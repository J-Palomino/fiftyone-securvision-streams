"""SecurVision Streams — FiftyOne plugin for live camera streaming.

Captures frames from RTSP, webcam, or synthetic demo sources in
background threads and displays them in a FiftyOne panel.  Snapshots
can be saved to the active dataset for review and labeling.

Usage (after installing):
    1. Open the FiftyOne App with a dataset loaded
    2. Open the "Camera Streams" panel (via the + tab or panel browser)
    3. Click "Add Camera" to connect an RTSP URL, webcam, or demo feed
    4. Frames auto-refresh every 2 seconds via the built-in timer
    5. Click "Snapshot All" to persist current frames to the dataset
"""

import atexit
import base64
import datetime
import logging
import os
import threading
import time

import cv2
import numpy as np
import requests

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

log = logging.getLogger(__name__)

# Directory for transient per-camera frame JPEGs (overwritten each cycle)
_FRAME_DIR = os.path.join(os.path.dirname(__file__), "_frames")


# ─── Stream Manager ───────────────────────────────────────────────────


class StreamManager:
    """Thread-safe background capture manager for live camera sources.

    Each camera runs in its own daemon thread, continuously grabbing
    frames via OpenCV.  The latest frame is kept in memory as both a
    raw numpy array and a base64-encoded JPEG for panel display.
    """

    # After this many consecutive read failures, attempt reconnection
    _RECONNECT_THRESHOLD = 50   # ~5 s at 10 fps
    _MAX_RECONNECT_ATTEMPTS = 5

    def __init__(self):
        self._cams = {}          # cam_id -> camera dict
        self._lock = threading.Lock()

    # -- public API --------------------------------------------------

    def add(self, cam_id, name, url, source_type="rtsp"):
        """Register a camera and start its capture thread."""
        with self._lock:
            if cam_id in self._cams:
                return False
            stop_event = threading.Event()
            self._cams[cam_id] = dict(
                id=cam_id, name=name, url=url, source_type=source_type,
                cap=None, frame=None, jpeg_b64=None, jpeg_path=None,
                status="connecting", stop=stop_event, ts=0,
            )
        t = threading.Thread(target=self._capture_loop, args=(cam_id,),
                             daemon=True, name=f"cam-{cam_id}")
        t.start()
        return True

    def remove(self, cam_id):
        """Signal the capture thread to stop and remove the camera.

        The thread owns its capture object and is responsible for
        releasing it — we never touch cap from outside the thread.
        """
        with self._lock:
            cam = self._cams.pop(cam_id, None)
        if not cam:
            return False
        # Signal the thread; it will release the capture and exit.
        cam["stop"].set()
        # Clean up transient frame file (best-effort).
        path = cam.get("jpeg_path")
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
        return True

    def shutdown(self):
        """Stop all cameras.  Called by the atexit hook."""
        with self._lock:
            cam_ids = list(self._cams.keys())
        for cid in cam_ids:
            self.remove(cid)

    def list_cameras(self):
        """Return a list of camera info dicts (no heavy data)."""
        with self._lock:
            return [
                dict(id=c["id"], name=c["name"], url=c["url"],
                     source_type=c["source_type"], status=c["status"])
                for c in self._cams.values()
            ]

    def frame_b64(self, cam_id):
        """Return the latest JPEG frame as a base64 string, or None."""
        with self._lock:
            c = self._cams.get(cam_id)
            return c["jpeg_b64"] if c else None

    def frame_numpy(self, cam_id):
        """Return a copy of the latest raw frame, or None."""
        with self._lock:
            c = self._cams.get(cam_id)
            if c and c["frame"] is not None:
                return c["frame"].copy()
        return None

    def snapshot(self, cam_id, dataset, out_dir="securvision_snapshots"):
        """Save the current frame to disk and add it to *dataset*."""
        frame = self.frame_numpy(cam_id)
        if frame is None:
            return None
        with self._lock:
            info = dict(self._cams[cam_id]) if cam_id in self._cams else None
        if info is None:
            return None

        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        path = os.path.abspath(
            os.path.join(out_dir, f"{cam_id}_{ts}.jpg"))
        cv2.imwrite(path, frame)

        sample = fo.Sample(filepath=path)
        sample["camera_id"] = cam_id
        sample["camera_name"] = info["name"]
        sample["source_url"] = info["url"]
        sample["captured_at"] = time.time()
        sample.tags = [cam_id, "snapshot"]
        dataset.add_sample(sample)
        return sample

    # -- internal ----------------------------------------------------

    def _capture_loop(self, cam_id):
        """Background thread: connect, grab frames, reconnect on failure."""
        with self._lock:
            cam = self._cams.get(cam_id)
        if not cam:
            return

        stop = cam["stop"]
        cap = None
        browser_only = cam["source_type"] == "hls"

        try:
            cap = self._connect(cam_id, cam)
            if cap is None:
                return

            # HLS sources are played in-browser — the thread just
            # keeps the camera "alive" so it appears in list_cameras().
            if browser_only:
                self._set_status(cam_id, "streaming")
                stop.wait()  # block until removal
                return

            os.makedirs(_FRAME_DIR, exist_ok=True)
            frame_path = os.path.join(_FRAME_DIR, f"{cam_id}.jpg")

            fail_count = 0
            reconnects = 0
            last_error_log = 0.0

            while not stop.is_set():
                try:
                    ret, frame = cap.read()
                except Exception as exc:
                    # Log at most once per 10 s to avoid spam.
                    now = time.time()
                    if now - last_error_log > 10:
                        log.warning("[%s] read error: %s", cam_id, exc)
                        last_error_log = now
                    ret, frame = False, None

                if ret and frame is not None:
                    fail_count = 0
                    reconnects = 0
                    _, buf = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                    cv2.imwrite(frame_path, frame)
                    with self._lock:
                        if cam_id in self._cams:
                            c = self._cams[cam_id]
                            c["frame"] = frame
                            c["jpeg_b64"] = b64
                            c["jpeg_path"] = os.path.abspath(frame_path)
                            c["status"] = "streaming"
                            c["ts"] = time.time()
                else:
                    fail_count += 1
                    self._set_status(cam_id, "no_frame")

                    if fail_count >= self._RECONNECT_THRESHOLD:
                        if reconnects >= self._MAX_RECONNECT_ATTEMPTS:
                            log.error(
                                "[%s] giving up after %d reconnect attempts",
                                cam_id, reconnects,
                            )
                            self._set_status(cam_id, "error")
                            return

                        reconnects += 1
                        fail_count = 0
                        log.info(
                            "[%s] reconnecting (attempt %d/%d)…",
                            cam_id, reconnects,
                            self._MAX_RECONNECT_ATTEMPTS,
                        )
                        self._set_status(cam_id, "reconnecting")
                        self._release_cap(cap)
                        # Back off before reconnecting.
                        if stop.wait(min(2 ** reconnects, 30)):
                            return  # stop was signalled
                        cap = self._connect(cam_id, cam)
                        if cap is None:
                            return

                stop.wait(0.1)   # ~10 fps capture ceiling
        finally:
            # Thread owns the capture — always release here.
            self._release_cap(cap)

    def _connect(self, cam_id, cam):
        """Open a capture source. Returns the capture or None on failure."""
        try:
            cap = self._open_capture(cam)
            opened = cap.isOpened() if hasattr(cap, "isOpened") else True
            if not opened:
                self._release_cap(cap)  # C3: don't leak the object
                self._set_status(cam_id, "error")
                log.error("[%s] cannot open %s", cam_id, cam["url"])
                return None
            with self._lock:
                if cam_id in self._cams:
                    self._cams[cam_id]["cap"] = cap
            self._set_status(cam_id, "connected")
            return cap
        except Exception as exc:
            self._set_status(cam_id, "error")
            log.error("[%s] connect error: %s", cam_id, exc)
            return None

    @staticmethod
    def _release_cap(cap):
        """Safely release an OpenCV capture (or no-op for None/demo)."""
        if cap is not None and hasattr(cap, "release"):
            try:
                cap.release()
            except Exception:
                pass

    def _open_capture(self, cam):
        src = cam["source_type"]
        url = cam["url"]

        if src == "demo":
            return _DemoCapture(cam["name"])

        if src == "webcam":
            idx = int(url) if url.isdigit() else 0
            return cv2.VideoCapture(idx)

        if src == "video":
            return _VideoFileCapture(url)

        if src == "http_snapshot":
            return _HttpSnapshotCapture(url)

        # HLS sources are played in-browser via MediaPlayerView — no
        # OpenCV capture needed.  We store a sentinel so the panel
        # knows to render a MediaPlayerView instead of an img().
        if src == "hls":
            return _BrowserOnlyCapture()

        # RTSP (default) — force TCP transport and set timeouts.
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Force TCP transport for reliability.  Many IP cameras drop
        # frames over UDP on congested / Wi-Fi networks.
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp",
        )
        return cap

    def _set_status(self, cam_id, status):
        with self._lock:
            if cam_id in self._cams:
                self._cams[cam_id]["status"] = status


# ─── Synthetic Demo Source ─────────────────────────────────────────────


class _DemoCapture:
    """Generates synthetic security-camera frames for testing."""

    def __init__(self, label="Demo", w=640, h=480):
        self.label = label
        self.w = w
        self.h = h
        self._n = 0
        self._rng = np.random.RandomState(abs(hash(label)) % (2 ** 31))

    def isOpened(self):
        return True

    def read(self):
        f = np.full((self.h, self.w, 3), (40, 42, 45), dtype=np.uint8)

        # Ground plane
        cv2.rectangle(f, (0, self.h // 2), (self.w, self.h),
                      (55, 58, 52), -1)
        cv2.line(f, (0, self.h // 2), (self.w, self.h // 2),
                 (70, 72, 68), 1)

        # Door (opens periodically)
        dx = self.w - 120
        door_open = (self._n // 30) % 5 == 0
        dc = (90, 130, 90) if not door_open else (50, 70, 50)
        cv2.rectangle(f, (dx, self.h // 2 - 100),
                      (dx + 60, self.h // 2), dc, -1)
        cv2.rectangle(f, (dx, self.h // 2 - 100),
                      (dx + 60, self.h // 2), (100, 100, 100), 1)
        if door_open:
            cv2.putText(f, "OPEN", (dx + 5, self.h // 2 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Moving actors (people + vehicle)
        actors = [
            ((60, 80, 200), 30, 70, "P"),
            ((80, 60, 180), 28, 65, "P"),
            ((180, 160, 40), 90, 50, "V"),
        ]
        for i, (color, aw, ah, tag) in enumerate(actors):
            t = ((self._n + i * 67) % 200) / 200
            x = int(t * (self.w - aw))
            y = self.h // 2 - ah + self._rng.randint(-5, 6)
            cv2.rectangle(f, (x, y), (x + aw, y + ah), color, -1)
            cv2.putText(f, tag, (x + 4, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Overlays: camera label + timestamp
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(f, self.label, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(f, ts, (10, self.h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        noise = self._rng.randint(0, 8, f.shape, dtype=np.uint8)
        f = cv2.add(f, noise)
        self._n += 1
        return True, f

    def release(self):
        pass


class _VideoFileCapture:
    """Wraps cv2.VideoCapture for a video file with loop-on-end."""

    def __init__(self, path):
        self.path = path
        self._cap = cv2.VideoCapture(path)

    def isOpened(self):
        return self._cap.isOpened()

    def read(self):
        ret, frame = self._cap.read()
        if not ret:
            # Loop back to the start of the file.
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
        return ret, frame

    def release(self):
        self._cap.release()


class _HttpSnapshotCapture:
    """Polls an HTTP JPEG endpoint on each read().

    Works with IP cameras that expose a snapshot URL such as
    ``http://192.168.1.100/snap.jpg``.  Each ``read()`` does a fresh
    HTTP GET and decodes the response as an image.
    """

    def __init__(self, url, timeout=5):
        self.url = url
        self.timeout = timeout

    def isOpened(self):
        try:
            r = requests.head(self.url, timeout=self.timeout)
            return r.status_code < 400
        except Exception:
            return False

    def read(self):
        try:
            r = requests.get(self.url, timeout=self.timeout)
            r.raise_for_status()
            arr = np.frombuffer(r.content, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                return True, frame
        except Exception:
            pass
        return False, None

    def release(self):
        pass


class _BrowserOnlyCapture:
    """Sentinel for sources played directly in the browser (HLS/DASH).

    No server-side frame decode happens — the panel renders a
    ``MediaPlayerView`` instead.  The capture thread stays alive so
    the camera appears in ``list_cameras()`` with status
    ``"streaming"``.
    """

    def isOpened(self):
        return True

    def read(self):
        # Yield no frames; display is browser-side.
        return False, None

    def release(self):
        pass


# ─── Module singleton ──────────────────────────────────────────────────

_mgr = StreamManager()
atexit.register(_mgr.shutdown)


# ─── Operators ─────────────────────────────────────────────────────────


class AddCamera(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="add_camera",
            label="Add Camera Stream",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str(
            "camera_id",
            label="Camera ID",
            required=True,
            description="Unique identifier (e.g. lobby-1)",
        )
        inputs.str(
            "name",
            label="Name",
            required=True,
            description="Display name (e.g. Lobby Entrance)",
        )
        inputs.str(
            "url",
            label="Stream URL",
            required=True,
            description=(
                "RTSP URL (rtsp://…), HLS URL (https://…/stream.m3u8), "
                "HTTP snapshot (http://cam/snap.jpg), video file path, "
                "webcam device index (0), or any string for demo"
            ),
        )
        inputs.enum(
            "source_type",
            values=types.Choices(choices=[
                types.Choice("rtsp", label="RTSP Stream (TCP)"),
                types.Choice("hls",
                             label="HLS / Web Stream (browser)"),
                types.Choice("http_snapshot",
                             label="HTTP Snapshot (periodic JPEG)"),
                types.Choice("video", label="Video File"),
                types.Choice("webcam", label="Local Webcam"),
                types.Choice("demo", label="Demo (synthetic)"),
            ]),
            default="rtsp",
            label="Source Type",
        )
        return types.Property(inputs, view=types.View(
            label="Add Camera Stream"))

    def execute(self, ctx):
        added = _mgr.add(
            ctx.params["camera_id"],
            ctx.params["name"],
            ctx.params["url"],
            ctx.params.get("source_type", "rtsp"),
        )
        return {"added": added}

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.bool("added", label="Camera added")
        return types.Property(outputs)


class RemoveCamera(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="remove_camera",
            label="Remove Camera Stream",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        cams = _mgr.list_cameras()
        if cams:
            inputs.enum(
                "camera_id",
                values=types.Choices(choices=[
                    types.Choice(c["id"],
                                 label=f"{c['name']} ({c['id']})")
                    for c in cams
                ]),
                label="Camera",
                required=True,
            )
        else:
            inputs.view(
                "notice",
                types.Notice(label="No cameras are connected"),
            )
        return types.Property(inputs, view=types.View(
            label="Remove Camera"))

    def execute(self, ctx):
        cam_id = ctx.params.get("camera_id")
        removed = _mgr.remove(cam_id) if cam_id else False
        return {"removed": removed}


class SnapshotCamera(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="snapshot_camera",
            label="Save Camera Snapshot",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        streaming = [
            c for c in _mgr.list_cameras() if c["status"] == "streaming"
        ]
        if streaming:
            inputs.enum(
                "camera_id",
                values=types.Choices(choices=[
                    types.Choice(c["id"], label=c["name"])
                    for c in streaming
                ]),
                label="Camera",
                required=True,
            )
        else:
            inputs.view(
                "notice",
                types.Notice(label="No cameras currently streaming"),
            )
        return types.Property(inputs, view=types.View(
            label="Save Snapshot"))

    def execute(self, ctx):
        cam_id = ctx.params.get("camera_id")
        if not cam_id or not ctx.dataset:
            return {"saved": False}
        sample = _mgr.snapshot(cam_id, ctx.dataset)
        if sample:
            ctx.trigger("reload_dataset")
            return {"saved": True, "filepath": sample.filepath}
        return {"saved": False}


class SnapshotAll(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="snapshot_all_cameras",
            label="Snapshot All Cameras",
        )

    def execute(self, ctx):
        if not ctx.dataset:
            return {"saved": 0}
        count = 0
        for cam in _mgr.list_cameras():
            if cam["status"] == "streaming":
                if _mgr.snapshot(cam["id"], ctx.dataset):
                    count += 1
        if count:
            ctx.trigger("reload_dataset")
        return {"saved": count}


class RefreshStreams(foo.Operator):
    """Called by the panel's TimerView to push fresh frames into state."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="refresh_streams",
            label="Refresh Camera Streams",
        )

    def execute(self, ctx):
        # Push latest base64 frames into panel state so img() picks them up
        cameras = _mgr.list_cameras()
        state_patch = {"_cameras": cameras}
        for cam in cameras:
            b64 = _mgr.frame_b64(cam["id"])
            if b64:
                state_patch[f"frame_{cam['id']}"] = (
                    f"data:image/jpeg;base64,{b64}"
                )
            else:
                state_patch[f"frame_{cam['id']}"] = None
        ctx.panel.state.set(state_patch)
        return {}


# ─── Panel ─────────────────────────────────────────────────────────────


class CameraStreamsPanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="camera_streams",
            label="Camera Streams",
            allow_multiple=False,
            surfaces="grid modal",
        )

    def on_load(self, ctx):
        # Seed state with current camera list + frames
        cameras = _mgr.list_cameras()
        state = {"_cameras": cameras}
        for cam in cameras:
            b64 = _mgr.frame_b64(cam["id"])
            if b64:
                state[f"frame_{cam['id']}"] = (
                    f"data:image/jpeg;base64,{b64}"
                )
        ctx.panel.state.set(state)

    def render(self, ctx):
        panel = types.Object()
        cameras = ctx.panel.state.get("_cameras") or _mgr.list_cameras()

        n_streaming = sum(
            1 for c in cameras if c.get("status") == "streaming"
        )

        # ── header ──
        panel.str(
            "header",
            default=(
                f"**{len(cameras)}** camera(s) connected — "
                f"**{n_streaming}** streaming"
            ),
            view=types.MarkdownView(),
        )

        # ── action buttons ──
        panel.btn(
            "add_btn",
            label="Add Camera",
            on_click="@securvision/streams/add_camera",
            variant="contained",
        )
        panel.btn(
            "snap_btn",
            label="Snapshot All",
            on_click="@securvision/streams/snapshot_all_cameras",
        )
        panel.btn(
            "snap_one_btn",
            label="Snapshot One",
            on_click="@securvision/streams/snapshot_camera",
        )
        panel.btn(
            "remove_btn",
            label="Remove Camera",
            on_click="@securvision/streams/remove_camera",
            variant="outlined",
        )

        if not cameras:
            panel.view(
                "empty",
                types.Notice(
                    label=(
                        "No cameras connected. Click Add Camera to "
                        "start streaming from an RTSP source, webcam, "
                        "or demo feed."
                    ),
                ),
            )
            return types.Property(panel)

        # ── camera feed cards ──
        for cam in cameras:
            cam_id = cam["id"]
            status = cam.get("status", "unknown")
            status_label = {
                "streaming": "streaming",
                "connected": "connected",
                "connecting": "connecting…",
                "reconnecting": "reconnecting…",
                "error": "ERROR",
                "no_frame": "no frame",
            }.get(status, status)

            panel.str(
                f"label_{cam_id}",
                default=(
                    f"**{cam['name']}** `{cam_id}` — "
                    f"*{status_label}*  \n`{cam['url']}`"
                ),
                view=types.MarkdownView(),
            )

            src_type = cam.get("source_type", "rtsp")

            if src_type == "hls":
                # HLS / web streams play directly in the browser —
                # no server-side decode, no base64 frames.
                panel.view(
                    f"player_{cam_id}",
                    types.MediaPlayerView(url=cam["url"]),
                )
            else:
                # OpenCV-decoded sources: show latest base64 frame.
                has_frame = (
                    ctx.panel.state.get(f"frame_{cam_id}") is not None
                )
                if has_frame:
                    panel.img(
                        f"frame_{cam_id}",
                        height="300px",
                        width="100%",
                    )
                else:
                    panel.view(
                        f"wait_{cam_id}",
                        types.Notice(
                            label=(
                                f"Waiting for first frame from "
                                f"{cam['name']}…"
                            ),
                        ),
                    )

        # ── auto-refresh timer ──
        # TimerView calls the refresh operator every 2 s, which pushes
        # updated base64 frames into panel state → triggers re-render.
        panel.view(
            "refresh_timer",
            types.TimerView(
                interval=2000,
                on_interval="@securvision/streams/refresh_streams",
            ),
        )

        # ── footer ──
        panel.str(
            "hint",
            default=(
                "---\n*Frames auto-refresh every 2 s. Use Snapshot All "
                "to save current frames to the dataset for labeling.*"
            ),
            view=types.MarkdownView(),
        )

        return types.Property(panel)


# ─── Registration ──────────────────────────────────────────────────────


def register(p):
    p.register(AddCamera)
    p.register(RemoveCamera)
    p.register(SnapshotCamera)
    p.register(SnapshotAll)
    p.register(RefreshStreams)
    p.register(CameraStreamsPanel)
