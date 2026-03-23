"""SecurVision Streams — FiftyOne plugin for live camera streaming.

Captures frames from RTSP, webcam, or synthetic demo sources in
background threads and displays them in a FiftyOne panel.  Snapshots
can be saved to the active dataset for review and labeling.

Features:
    - Live streaming from RTSP, HLS, HTTP snapshot, webcam, video, demo
    - ONVIF camera auto-discovery
    - Motion detection with configurable sensitivity
    - MP4 recording (continuous or motion-triggered with pre/post-roll)
    - Email alerts on motion with cooldown

Usage (after installing):
    1. Open the FiftyOne App with a dataset loaded
    2. Open the "Camera Streams" panel (via the + tab or panel browser)
    3. Click "Add Camera" to connect an RTSP URL, webcam, or demo feed
    4. Frames auto-refresh every 2 seconds via the built-in timer
    5. Click "Snapshot All" to persist current frames to the dataset
"""

import atexit
import base64
import collections
import datetime
import email.mime.image
import email.mime.multipart
import email.mime.text
import json
import logging
import os
import queue
import re
import smtplib
import threading
import time
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

log = logging.getLogger(__name__)

# Force TCP transport for RTSP streams process-wide.  Many IP cameras drop
# frames over UDP on congested / Wi-Fi networks.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


# ─── Default Configurations ──────────────────────────────────────────

_default_motion_config = {
    "threshold": 25,       # pixel diff threshold for cv2.threshold
    "min_area": 500,       # minimum contour area to count as motion
    "enabled": True,
}

_default_recording_config = {
    "output_dir": "securvision_recordings",
    "segment_seconds": 300,    # 5-minute file segments
    "preroll_seconds": 5,
    "postroll_seconds": 10,
    "fps": 10,
}

_alert_config = {
    "enabled": False,
    "smtp_host": "",
    "smtp_port": 587,
    "smtp_tls": True,
    "sender": "",
    "password": "",
    "recipients": [],
    "cooldown_seconds": 300,
}


# ─── Persistent Config ────────────────────────────────────────────────


def _config_path():
    return os.path.join(os.path.dirname(__file__), "securvision_config.json")


def _load_config():
    """Read saved config from JSON. Returns dict or empty dict."""
    path = _config_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("Failed to load config from %s: %s", path, exc)
        return {}


def _save_config():
    """Persist current camera + global config to JSON."""
    # Import here to avoid circular ref at module load
    cameras = {}
    try:
        with _mgr._lock:
            for cam_id, c in _mgr._cams.items():
                cameras[cam_id] = {
                    "name": c["name"], "url": c["url"],
                    "source_type": c["source_type"],
                    "region": c.get("region", ""),
                    "locale": c.get("locale", ""),
                    "room": c.get("room", ""),
                    "position": c.get("position", ""),
                    "notes": c.get("notes", ""),
                    "custom_zones": c.get("custom_zones", []),
                    "zones": c.get("zones",
                                   [[True, True, True] for _ in range(3)]),
                    "zone_names": c.get("zone_names", []),
                    "motion_config": dict(c.get("motion_config", {})),
                    "recording_config": dict(c.get("recording_config", {})),
                    "recording_mode": c.get("recording_mode", "off"),
                    "alerts_enabled": c.get("alerts_enabled", True),
                }
    except Exception:
        cameras = {}
    cfg = {
        "cameras": cameras,
        "alerts": {k: v for k, v in _alert_config.items()},
        "detection": {k: v for k, v in _detection_config.items()},
    }
    try:
        with open(_config_path(), "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as exc:
        log.error("Failed to save config: %s", exc)


def _auto_reconnect():
    """Load saved config and reconnect all cameras on startup."""
    cfg = _load_config()
    if not cfg:
        return
    # Restore global configs
    if "alerts" in cfg:
        _alert_config.update(cfg["alerts"])
    if "detection" in cfg:
        _detection_config.update(cfg["detection"])
    # Reconnect cameras
    for cam_id, cc in cfg.get("cameras", {}).items():
        try:
            _mgr.add(
                cam_id, cc.get("name", cam_id), cc.get("url", ""),
                cc.get("source_type", "rtsp"),
                region=cc.get("region", ""),
                locale=cc.get("locale", ""),
                room=cc.get("room", ""),
                position=cc.get("position", ""),
                notes=cc.get("notes", ""),
            )
            # Restore zones (grid + custom)
            if "zones" in cc or "zone_names" in cc or "custom_zones" in cc:
                with _mgr._lock:
                    cam = _mgr._cams.get(cam_id)
                    if cam:
                        if "zones" in cc:
                            cam["zones"] = cc["zones"]
                        if "zone_names" in cc:
                            cam["zone_names"] = cc["zone_names"]
                        if "custom_zones" in cc:
                            cam["custom_zones"] = cc["custom_zones"]
                        cam["_zone_mask"] = None
            if "motion_config" in cc:
                _mgr.configure_motion(cam_id, **cc["motion_config"])
            mode = cc.get("recording_mode", "off")
            if mode != "off":
                _mgr.configure_recording(cam_id, mode=mode)
            rec_cfg = cc.get("recording_config")
            if rec_cfg:
                _mgr.configure_recording(cam_id, **rec_cfg)
            if "alerts_enabled" in cc:
                _mgr.set_alerts_enabled(cam_id, cc["alerts_enabled"])
        except Exception as exc:
            log.warning("Failed to reconnect camera %s: %s", cam_id, exc)


_detection_config = {
    "model_name": "yolov8s-coco-torch",
    "seg_model_name": "yolov8s-seg-coco-torch",
    "confidence_threshold": 0.25,
    "label_field": "detections",
    "auto_detect_snapshots": False,
    "overlay_enabled": False,
    "overlay_interval": 5,  # run inference every N frames
    "overlay_mode": "box",  # "box" (fast rectangles) or "segmentation" (organic shapes)
    "seg_opacity": 0.4,     # overlay opacity for segmentation masks
}


# ─── Detection Model Management ──────────────────────────────────────

_detection_available = None
_detection_model = None
_detection_model_name = None
_model_lock = threading.Lock()


def _check_detection_deps():
    """Return True if fiftyone.zoo and torch are importable."""
    global _detection_available
    if _detection_available is not None:
        return _detection_available
    try:
        import fiftyone.zoo  # noqa: F401
        _detection_available = True
    except ImportError:
        _detection_available = False
    return _detection_available


def _get_detection_model(model_name=None):
    """Lazy-load and cache a detection model. Thread-safe."""
    global _detection_model, _detection_model_name
    name = model_name or _detection_config["model_name"]
    with _model_lock:
        if _detection_model is not None and _detection_model_name == name:
            return _detection_model
        import fiftyone.zoo as foz
        _detection_model = foz.load_zoo_model(name)
        _detection_model_name = name
        log.info("Loaded detection model: %s", name)
        return _detection_model


def _label_color(label):
    """Deterministic BGR color per detection label for overlay boxes."""
    colors = {
        "person": (0, 255, 0), "car": (255, 0, 0),
        "truck": (255, 128, 0), "bus": (255, 100, 50),
        "dog": (0, 255, 255), "cat": (0, 165, 255),
        "bicycle": (200, 200, 0), "motorcycle": (200, 100, 200),
    }
    if label in colors:
        return colors[label]
    h = abs(hash(label)) % 180
    bgr = cv2.cvtColor(
        np.uint8([[[h, 200, 220]]]), cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def _run_detection_on_image(image, model=None):
    """Run detection on a numpy image (BGR from OpenCV), return fo.Detections."""
    if model is None:
        model = _get_detection_model()
    # FiftyOne zoo models expect RGB; OpenCV frames are BGR
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = model.predict(rgb)
    # Filter by confidence threshold
    thresh = _detection_config["confidence_threshold"]
    if hasattr(detections, "detections") and thresh > 0:
        detections.detections = [
            d for d in detections.detections
            if (d.confidence or 0) >= thresh
        ]
    return detections


# ─── Recording Index Infrastructure ──────────────────────────────────

_recording_index_queue = None
_recording_indexer_thread = None
_indexer_dataset_name = None


def _ensure_indexer_running(dataset_name):
    """Start the recording indexer thread if not already running."""
    global _recording_index_queue, _recording_indexer_thread
    global _indexer_dataset_name
    _indexer_dataset_name = dataset_name
    if _recording_index_queue is None:
        _recording_index_queue = queue.Queue(maxsize=1000)
    if (_recording_indexer_thread is None
            or not _recording_indexer_thread.is_alive()):
        _recording_indexer_thread = threading.Thread(
            target=_recording_indexer_loop, daemon=True,
            name="recording-indexer")
        _recording_indexer_thread.start()


def _recording_indexer_loop():
    """Drain the recording index queue and add segments to the dataset."""
    dataset = None
    current_name = None
    while True:
        try:
            meta = _recording_index_queue.get(timeout=5)
        except queue.Empty:
            continue
        try:
            # Reload dataset if name changed or dataset ref is stale
            name = _indexer_dataset_name
            if name and (dataset is None or current_name != name):
                dataset = fo.load_dataset(name)
                current_name = name
            if dataset is None:
                continue
            sample = fo.Sample(filepath=meta["path"])
            sample["camera_id"] = meta["camera_id"]
            sample["camera_name"] = meta["camera_name"]
            sample["source_url"] = meta["source_url"]
            sample["region"] = meta.get("region", "")
            sample["locale"] = meta.get("locale", "")
            sample["room"] = meta.get("room", "")
            sample["position"] = meta.get("position", "")
            sample["notes"] = meta.get("notes", "")
            sample["started_at"] = meta["started_at"]
            sample["ended_at"] = meta["ended_at"]
            sample["captured_at"] = meta["started_at"]
            sample["duration_seconds"] = meta["duration_seconds"]
            sample["recording_mode"] = meta["recording_mode"]
            sample["sample_type"] = "recording"
            sample.tags = [meta["camera_id"], "recording"]
            dataset.add_sample(sample)
        except Exception as exc:
            log.error("Recording indexer error: %s", exc)
            dataset = None  # force reload on next iteration


# ─── Email Alert Helper ──────────────────────────────────────────────


def _send_alert_email(cam_id, cam_name, score, jpeg_bytes,
                      active_zones=None, notes=""):
    """Send a motion alert email with attached snapshot.

    Runs in a short-lived daemon thread to avoid blocking the capture loop.
    """
    cfg = _alert_config
    if not cfg["enabled"] or not cfg["smtp_host"] or not cfg["recipients"]:
        return
    try:
        msg = email.mime.multipart.MIMEMultipart()
        msg["From"] = cfg["sender"]
        msg["To"] = ", ".join(cfg["recipients"])
        msg["Subject"] = (
            f"[SecurVision] Motion detected: {cam_name} ({cam_id})")

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        zones_str = (", ".join(active_zones)
                     if active_zones else "full frame")
        body = (
            f"Motion detected on camera '{cam_name}' (ID: {cam_id})\n"
            f"Time: {ts}\n"
            f"Motion score: {score:.2%}\n"
            f"Active zones: {zones_str}\n"
        )
        if notes:
            body += f"Camera notes: {notes}\n"
        msg.attach(email.mime.text.MIMEText(body, "plain"))

        if jpeg_bytes:
            img_part = email.mime.image.MIMEImage(
                jpeg_bytes, name=f"{cam_id}_motion.jpg")
            msg.attach(img_part)

        with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as smtp:
            if cfg["smtp_tls"]:
                smtp.starttls()
            if cfg["sender"] and cfg["password"]:
                smtp.login(cfg["sender"], cfg["password"])
            smtp.sendmail(cfg["sender"], cfg["recipients"], msg.as_string())

        log.info("[%s] Motion alert email sent to %s", cam_id,
                 cfg["recipients"])
    except Exception as exc:
        log.error("[%s] Failed to send alert email: %s", cam_id, exc)


# ─── Stream Manager ───────────────────────────────────────────────────


class StreamManager:
    """Thread-safe background capture manager for live camera sources.

    Each camera runs in its own daemon thread, continuously grabbing
    frames via OpenCV.  The latest frame is kept in memory as both a
    raw numpy array and a base64-encoded JPEG for panel display.

    Motion detection, recording, and alerts are processed inline in each
    camera's capture thread.
    """

    _RECONNECT_THRESHOLD = 50   # ~5 s at 10 fps
    _MAX_RECONNECT_ATTEMPTS = 5

    def __init__(self):
        self._cams = {}
        self._lock = threading.Lock()

    # -- public API --------------------------------------------------

    def add(self, cam_id, name, url, source_type="rtsp",
            region="", locale="", room="", position="", notes=""):
        """Register a camera and start its capture thread."""
        with self._lock:
            if cam_id in self._cams:
                return False
            stop_event = threading.Event()
            self._cams[cam_id] = dict(
                id=cam_id, name=name, url=url, source_type=source_type,
                region=region, locale=locale, room=room, position=position,
                notes=notes,
                cap=None, frame=None, display_frame=None, jpeg_b64=None,
                status="connecting", stop=stop_event, thread=None, ts=0,
                # Motion detection
                motion_detected=False, motion_score=0.0,
                last_motion_ts=0.0,
                motion_config=dict(_default_motion_config),
                _prev_gray=None,
                # Recording
                recording_active=False,
                recording_mode="off",
                recording_config=dict(_default_recording_config),
                _writer=None, _segment_start=0.0,
                _current_segment_path=None,
                _preroll_buffer=None, _postroll_deadline=0.0,
                # Zones (3x3 grid)
                zones=[[True]*3, [True]*3, [True]*3],
                zone_names=[
                    ["top-left", "top-center", "top-right"],
                    ["mid-left", "center", "mid-right"],
                    ["bottom-left", "bottom-center", "bottom-right"],
                ],
                custom_zones=[],  # list of {name, x, y, w, h} (normalized)
                active_zones=[], _zone_mask=None,
                detection_counts={},  # e.g. {"person": 3, "car": 2}
                _last_detections=None, _detect_frame_counter=0,
                # Alerts
                alerts_enabled=True, _last_alert_ts=0.0,
            )
        t = threading.Thread(target=self._capture_loop, args=(cam_id,),
                             daemon=True, name=f"cam-{cam_id}")
        with self._lock:
            if cam_id in self._cams:
                self._cams[cam_id]["thread"] = t
        t.start()
        return True

    def remove(self, cam_id):
        """Signal the capture thread to stop and remove the camera."""
        with self._lock:
            cam = self._cams.pop(cam_id, None)
        if not cam:
            return False
        cam["stop"].set()
        t = cam.get("thread")
        if t is not None:
            t.join(timeout=3)
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
                dict(
                    id=c["id"], name=c["name"], url=c["url"],
                    source_type=c["source_type"], status=c["status"],
                    region=c.get("region", ""),
                    locale=c.get("locale", ""),
                    room=c.get("room", ""),
                    position=c.get("position", ""),
                    notes=c.get("notes", ""),
                    active_zones=c.get("active_zones", []),
                    detection_counts=c.get("detection_counts", {}),
                    motion_detected=c.get("motion_detected", False),
                    motion_score=c.get("motion_score", 0.0),
                    last_motion_ts=c.get("last_motion_ts", 0.0),
                    recording_active=c.get("recording_active", False),
                    recording_mode=c.get("recording_mode", "off"),
                    alerts_enabled=c.get("alerts_enabled", True),
                )
                for c in self._cams.values()
            ]

    def frame_b64(self, cam_id):
        """Return the latest JPEG frame as a base64 string, or None."""
        with self._lock:
            c = self._cams.get(cam_id)
            return c["jpeg_b64"] if c else None

    def grid_frame_b64(self, max_cols=4, cell_w=320, cell_h=240):
        """Build a composite grid image of all cameras. Returns b64 str."""
        # Copy frame refs + metadata under lock, then process outside
        cam_data = []
        with self._lock:
            for c in self._cams.values():
                f = c.get("display_frame")
                if f is None:
                    f = c.get("frame")
                cam_data.append((
                    f.copy() if f is not None else None,
                    c.get("name", c["id"]),
                    c.get("status", "?"),
                ))
        # Build grid frames outside lock
        frames = []
        for f, name, status in cam_data:
            if f is not None:
                resized = cv2.resize(f, (cell_w, cell_h))
                cv2.putText(resized, name, (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1)
                frames.append(resized)
            else:
                ph = np.full((cell_h, cell_w, 3), 30, dtype=np.uint8)
                cv2.putText(ph, name, (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (100, 100, 100), 1)
                cv2.putText(ph, status, (4, cell_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (80, 80, 80), 1)
                frames.append(ph)
        if not frames:
            return None
        n = len(frames)
        cols = min(n, max_cols)
        rows = (n + cols - 1) // cols
        grid_rows = []
        for r in range(rows):
            row_f = frames[r * cols:(r + 1) * cols]
            while len(row_f) < cols:
                row_f.append(np.full(
                    (cell_h, cell_w, 3), 20, dtype=np.uint8))
            grid_rows.append(np.hstack(row_f))
        grid = np.vstack(grid_rows)
        ok, buf = cv2.imencode(".jpg", grid,
                              [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok or buf is None:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")

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
            c = self._cams.get(cam_id)
            if c is None:
                return None
            info = {
                "name": c["name"], "url": c["url"],
                "region": c.get("region", ""),
                "locale": c.get("locale", ""),
                "room": c.get("room", ""),
                "position": c.get("position", ""),
                "notes": c.get("notes", ""),
                "active_zones": list(c.get("active_zones", [])),
                "motion_detected": c.get("motion_detected", False),
                "motion_score": c.get("motion_score", 0.0),
                "recording_active": c.get("recording_active", False),
            }

        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        path = os.path.abspath(
            os.path.join(out_dir, f"{cam_id}_{ts}.jpg"))
        if not cv2.imwrite(path, frame):
            log.error("Failed to write snapshot to %s", path)
            return None

        sample = fo.Sample(filepath=path)
        sample["camera_id"] = cam_id
        sample["camera_name"] = info["name"]
        sample["source_url"] = info["url"]
        sample["captured_at"] = datetime.datetime.fromtimestamp(time.time())
        sample["region"] = info["region"]
        sample["locale"] = info["locale"]
        sample["room"] = info["room"]
        sample["position"] = info["position"]
        sample["notes"] = info["notes"]
        sample["active_zones"] = info.get("active_zones", [])
        sample["motion_detected"] = info.get("motion_detected", False)
        sample["motion_score"] = info.get("motion_score", 0.0)
        sample["sample_type"] = "snapshot"
        sample.tags = [cam_id, "snapshot"]
        if info.get("motion_detected"):
            sample.tags.append("motion")
        if info.get("recording_active"):
            sample.tags.append("recording")
        dataset.add_sample(sample)

        # Optional: run object detection on the snapshot
        if _detection_config.get("auto_detect_snapshots") and frame is not None:
            try:
                model = _get_detection_model()
                dets = _run_detection_on_image(frame, model)
                sample[_detection_config["label_field"]] = dets
                # Compute and store detection counts
                counts = {}
                if hasattr(dets, "detections"):
                    for d in dets.detections:
                        counts[d.label] = counts.get(d.label, 0) + 1
                sample["detection_counts"] = counts
                sample.save()
                # Update cam dict so panel shows counts
                with self._lock:
                    c = self._cams.get(cam_id)
                    if c:
                        c["detection_counts"] = counts
            except Exception as exc:
                log.warning("[%s] Auto-detect on snapshot failed: %s",
                            cam_id, exc)

        return sample

    # -- configuration -----------------------------------------------

    def configure_motion(self, cam_id, **kwargs):
        """Update motion detection settings for a camera."""
        with self._lock:
            cam = self._cams.get(cam_id)
            if cam:
                cam["motion_config"].update(kwargs)
                return True
        return False

    def configure_recording(self, cam_id, **kwargs):
        """Update recording settings for a camera."""
        with self._lock:
            cam = self._cams.get(cam_id)
            if cam:
                if "mode" in kwargs:
                    cam["recording_mode"] = kwargs.pop("mode")
                cam["recording_config"].update(kwargs)
                # Resize pre-roll buffer if preroll/fps changed.
                cfg = cam["recording_config"]
                new_maxlen = max(1, int(
                    cfg["preroll_seconds"] * cfg["fps"]))
                old_buf = cam.get("_preroll_buffer")
                if old_buf is None or old_buf.maxlen != new_maxlen:
                    cam["_preroll_buffer"] = collections.deque(
                        maxlen=new_maxlen)
                return True
        return False

    def enable_recording_index(self, dataset_name):
        """Start the background recording indexer for a dataset."""
        _ensure_indexer_running(dataset_name)

    def set_alerts_enabled(self, cam_id, enabled):
        with self._lock:
            cam = self._cams.get(cam_id)
            if cam:
                cam["alerts_enabled"] = enabled
                return True
        return False

    # -- internal: capture loop --------------------------------------

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

            if browser_only:
                self._set_status(cam_id, "streaming")
                stop.wait()
                return

            # Initialize pre-roll buffer (only if not already set by config)
            if cam.get("_preroll_buffer") is None:
                preroll = cam["recording_config"]["preroll_seconds"]
                fps = cam["recording_config"]["fps"]
                cam["_preroll_buffer"] = collections.deque(
                    maxlen=max(1, int(preroll * fps)))

            fail_count = 0
            reconnects = 0
            last_error_log = 0.0

            while not stop.is_set():
                try:
                    ret, frame = cap.read()
                except Exception as exc:
                    now = time.time()
                    if now - last_error_log > 10:
                        log.warning("[%s] read error: %s", cam_id, exc)
                        last_error_log = now
                    ret, frame = False, None

                if ret and frame is not None:
                    fail_count = 0
                    reconnects = 0

                    # Motion detection (on raw frame)
                    motion, score = self._detect_motion(cam_id, frame, cam)

                    # Recording (writes raw frame, no overlay)
                    self._process_recording(cam_id, frame, cam, motion)

                    # Email alerts
                    if motion:
                        self._maybe_send_alert(cam_id, frame, cam)

                    # Live detection overlay (draws on copy)
                    display = frame
                    if (_detection_config.get("overlay_enabled")
                            and _check_detection_deps()):
                        display = self._apply_overlay(
                            cam_id, frame, cam)

                    ok, buf = cv2.imencode(
                        ".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if not ok or buf is None:
                        stop.wait(0.1)
                        continue
                    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

                    with self._lock:
                        if cam_id in self._cams:
                            c = self._cams[cam_id]
                            c["frame"] = frame
                            c["display_frame"] = display
                            c["jpeg_b64"] = b64
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
                        # Reset motion baseline so first frame after
                        # reconnect doesn't false-trigger.
                        cam["_prev_gray"] = None
                        if stop.wait(min(2 ** reconnects, 30)):
                            return
                        cap = self._connect(cam_id, cam)
                        if cap is None:
                            return

                stop.wait(0.1)   # ~10 fps capture ceiling
        finally:
            self._close_writer(cam)
            self._release_cap(cap)

    # -- internal: motion detection ----------------------------------

    @staticmethod
    def _build_zone_mask(h, w, zones):
        """Build a binary mask from the 3x3 zone grid."""
        mask = np.zeros((h, w), dtype=np.uint8)
        cell_h, cell_w = h // 3, w // 3
        for row in range(3):
            for col in range(3):
                if zones[row][col]:
                    y1 = row * cell_h
                    y2 = (row + 1) * cell_h if row < 2 else h
                    x1 = col * cell_w
                    x2 = (col + 1) * cell_w if col < 2 else w
                    mask[y1:y2, x1:x2] = 255
        return mask

    @staticmethod
    def _build_custom_zone_mask(h, w, custom_zones):
        """Build a binary mask from named rectangle zones."""
        mask = np.zeros((h, w), dtype=np.uint8)
        for z in custom_zones:
            x1 = max(0, int(z["x"] * w))
            y1 = max(0, int(z["y"] * h))
            x2 = min(w, int((z["x"] + z["w"]) * w))
            y2 = min(h, int((z["y"] + z["h"]) * h))
            mask[y1:y2, x1:x2] = 255
        return mask

    def _detect_motion(self, cam_id, frame, cam):
        """Frame differencing motion detection with zone masking."""
        cfg = cam["motion_config"]
        if not cfg.get("enabled", True):
            cam["motion_detected"] = False
            cam["motion_score"] = 0.0
            cam["active_zones"] = []
            return False, 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        prev_gray = cam.get("_prev_gray")
        cam["_prev_gray"] = gray

        if prev_gray is None:
            cam["motion_detected"] = False
            cam["motion_score"] = 0.0
            cam["active_zones"] = []
            return False, 0.0

        delta = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(
            delta, cfg["threshold"], 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Apply zone mask
        custom_zones = cam.get("custom_zones") or []
        use_custom = len(custom_zones) > 0
        fh, fw = frame.shape[:2]

        if use_custom:
            mask = cam.get("_zone_mask")
            if mask is None or mask.shape != (fh, fw):
                mask = self._build_custom_zone_mask(fh, fw, custom_zones)
                cam["_zone_mask"] = mask
            thresh = cv2.bitwise_and(thresh, mask)
        else:
            zones = cam.get("zones")
            if zones and not all(all(row) for row in zones):
                mask = cam.get("_zone_mask")
                if mask is None or mask.shape != (fh, fw):
                    mask = self._build_zone_mask(fh, fw, zones)
                    cam["_zone_mask"] = mask
                thresh = cv2.bitwise_and(thresh, mask)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = cfg["min_area"]

        motion_area = 0
        active = set()
        for c in contours:
            a = cv2.contourArea(c)
            if a >= min_area:
                motion_area += a
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    if use_custom:
                        # Check which custom zone the centroid falls in
                        for z in custom_zones:
                            zx1 = z["x"] * fw
                            zy1 = z["y"] * fh
                            zx2 = (z["x"] + z["w"]) * fw
                            zy2 = (z["y"] + z["h"]) * fh
                            if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                                active.add(z["name"])
                    else:
                        # 3x3 grid lookup
                        zone_names = cam.get("zone_names") or [
                            ["top-left", "top-center", "top-right"],
                            ["mid-left", "center", "mid-right"],
                            ["bottom-left", "bottom-center",
                             "bottom-right"],
                        ]
                        if fw > 0 and fh > 0:
                            col = max(0, min(
                                int(cx / (fw / 3.0)), 2))
                            row = max(0, min(
                                int(cy / (fh / 3.0)), 2))
                            active.add(zone_names[row][col])

        frame_area = fh * fw
        score = motion_area / frame_area if frame_area else 0.0

        detected = motion_area > 0
        cam["motion_detected"] = detected
        cam["motion_score"] = round(score, 4)
        cam["active_zones"] = sorted(active)
        if detected:
            cam["last_motion_ts"] = time.time()

        return detected, score

    # -- internal: detection overlay ----------------------------------

    def _apply_overlay(self, cam_id, frame, cam):
        """Run detection every N frames, draw cached results on every frame.

        Supports two modes via _detection_config["overlay_mode"]:
          "box"          — fast axis-aligned rectangles (default)
          "segmentation" — organic per-pixel masks tracing object shapes
        """
        cam["_detect_frame_counter"] = cam.get(
            "_detect_frame_counter", 0) + 1
        interval = _detection_config.get("overlay_interval", 5)
        use_seg = _detection_config.get("overlay_mode") == "segmentation"

        if cam["_detect_frame_counter"] >= interval:
            cam["_detect_frame_counter"] = 0
            try:
                if use_seg:
                    model_name = _detection_config.get(
                        "seg_model_name", "yolov8s-seg-coco-torch")
                else:
                    model_name = None  # uses default detection model
                dets = _run_detection_on_image(frame,
                                               _get_detection_model(model_name))
                cam["_last_detections"] = dets
                if hasattr(dets, "detections"):
                    counts = {}
                    for d in dets.detections:
                        counts[d.label] = counts.get(d.label, 0) + 1
                    cam["detection_counts"] = counts
            except Exception as exc:
                log.debug("[%s] overlay detection failed: %s",
                          cam_id, exc)

        dets = cam.get("_last_detections")
        if not dets or not hasattr(dets, "detections") or not dets.detections:
            return frame

        annotated = frame.copy()
        h, w = annotated.shape[:2]
        opacity = _detection_config.get("seg_opacity", 0.4)

        for det in dets.detections:
            bb = det.bounding_box
            if not bb or len(bb) < 4:
                continue
            x1, y1 = int(bb[0] * w), int(bb[1] * h)
            x2 = int((bb[0] + bb[2]) * w)
            y2 = int((bb[1] + bb[3]) * h)
            color = _label_color(det.label)

            # Segmentation mask overlay (organic shape)
            mask = getattr(det, "mask", None)
            if use_seg and mask is not None:
                # mask is a bool array sized to the bounding box region
                mh, mw = mask.shape[:2]
                bw, bh = x2 - x1, y2 - y1
                if mh > 0 and mw > 0 and bw > 0 and bh > 0:
                    # Resize mask to match bounding box in frame
                    resized = cv2.resize(
                        mask.astype(np.uint8) * 255, (bw, bh),
                        interpolation=cv2.INTER_NEAREST)
                    # Clamp to frame bounds
                    fx1 = max(0, x1)
                    fy1 = max(0, y1)
                    fx2 = min(w, x2)
                    fy2 = min(h, y2)
                    # Crop mask to match clamped region
                    mx1 = fx1 - x1
                    my1 = fy1 - y1
                    mx2 = mx1 + (fx2 - fx1)
                    my2 = my1 + (fy2 - fy1)
                    roi_mask = resized[my1:my2, mx1:mx2]
                    # Apply colored semi-transparent overlay
                    roi = annotated[fy1:fy2, fx1:fx2]
                    color_layer = np.full_like(roi, color, dtype=np.uint8)
                    blended = cv2.addWeighted(
                        color_layer, opacity, roi, 1 - opacity, 0)
                    mask_bool = roi_mask > 127
                    if mask_bool.shape[:2] == roi.shape[:2]:
                        roi[mask_bool] = blended[mask_bool]
                    # Draw contour outline around mask edge
                    contours, _ = cv2.findContours(
                        roi_mask, cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        cnt[:, :, 0] += fx1
                        cnt[:, :, 1] += fy1
                        cv2.drawContours(
                            annotated, [cnt], -1, color, 2)
            else:
                # Box mode — simple rectangle
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label text
            txt = det.label
            if det.confidence:
                txt += f" {det.confidence:.0%}"
            cv2.putText(annotated, txt, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return annotated

    # -- internal: recording -----------------------------------------

    def _process_recording(self, cam_id, frame, cam, motion_detected):
        """Handle recording: pre-roll buffer, start/stop, segment rotation."""
        cfg = cam["recording_config"]
        mode = cam.get("recording_mode", "off")

        if mode == "off":
            # Close any active writer left from a previous mode.
            if cam.get("_writer") is not None:
                self._close_writer(cam)
            return

        now = time.time()
        writer = cam.get("_writer")

        # Determine if we should be recording right now
        should_record = False
        if mode == "continuous":
            should_record = True
        elif mode == "motion":
            if motion_detected:
                cam["_postroll_deadline"] = now + cfg["postroll_seconds"]
                should_record = True
            elif now < cam.get("_postroll_deadline", 0):
                should_record = True

        # Start recording — flush pre-roll buffer into new file
        if should_record and writer is None:
            writer = self._open_writer(cam_id, cam, frame)
            if writer is not None:
                buf = cam.get("_preroll_buffer")
                if buf:
                    for buffered in buf:
                        try:
                            writer.write(buffered)
                        except Exception:
                            log.error("[%s] Pre-roll write failed",
                                      cam_id)
                            self._close_writer(cam)
                            return
                    buf.clear()

        # Write current frame
        if should_record and writer is not None:
            try:
                writer.write(frame)
            except Exception:
                log.error("[%s] Frame write failed, closing recording",
                          cam_id)
                self._close_writer(cam)
                cam["recording_active"] = False
                return
            # Segment rotation
            elapsed = now - cam.get("_segment_start", now)
            if elapsed >= cfg["segment_seconds"]:
                self._close_writer(cam)
                writer = self._open_writer(cam_id, cam, frame)

        # Stop recording (motion mode, post-roll expired)
        if not should_record and writer is not None:
            self._close_writer(cam)

        # Feed pre-roll buffer when in motion mode but not actively recording
        if not should_record and mode == "motion":
            buf = cam.get("_preroll_buffer")
            if buf is not None:
                buf.append(frame.copy())

        cam["recording_active"] = cam.get("_writer") is not None

    def _open_writer(self, cam_id, cam, frame):
        """Open a new cv2.VideoWriter for an MP4 segment."""
        cfg = cam["recording_config"]
        out_dir = cfg["output_dir"]
        os.makedirs(out_dir, exist_ok=True)

        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{cam_id}_{ts_str}.mp4"
        path = os.path.abspath(os.path.join(out_dir, filename))

        h, w = frame.shape[:2]
        fps = cfg["fps"]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

        if not writer.isOpened():
            log.error("[%s] Failed to open VideoWriter at %s", cam_id, path)
            return None

        cam["_writer"] = writer
        cam["_segment_start"] = time.time()
        cam["_current_segment_path"] = path
        log.info("[%s] Recording started: %s", cam_id, path)
        return writer

    def _close_writer(self, cam):
        """Release the current VideoWriter and enqueue for indexing."""
        writer = cam.get("_writer")
        if writer is None:
            return
        segment_path = cam.get("_current_segment_path")
        segment_start = cam.get("_segment_start", 0)
        try:
            writer.release()
        except Exception:
            pass
        cam["_writer"] = None
        cam["_current_segment_path"] = None
        cam["recording_active"] = False
        # Enqueue for dataset indexing (non-blocking)
        if segment_path and _recording_index_queue is not None:
            try:
                now = datetime.datetime.now()
                started = (datetime.datetime.fromtimestamp(segment_start)
                           if segment_start else now)
                duration = (time.time() - segment_start
                            if segment_start else 0)
                _recording_index_queue.put_nowait({
                    "path": segment_path,
                    "camera_id": cam.get("id", ""),
                    "camera_name": cam.get("name", ""),
                    "source_url": cam.get("url", ""),
                    "region": cam.get("region", ""),
                    "locale": cam.get("locale", ""),
                    "room": cam.get("room", ""),
                    "position": cam.get("position", ""),
                    "notes": cam.get("notes", ""),
                    "started_at": started,
                    "ended_at": now,
                    "duration_seconds": duration,
                    "recording_mode": cam.get("recording_mode", "off"),
                })
            except queue.Full:
                log.warning("Recording index queue full, segment skipped")

    # -- internal: alerts --------------------------------------------

    def _maybe_send_alert(self, cam_id, frame, cam):
        """Check cooldown and fire off email alert in background thread."""
        if not cam.get("alerts_enabled", True):
            return
        if not _alert_config.get("enabled", False):
            return

        now = time.time()
        cooldown = _alert_config.get("cooldown_seconds", 300)
        if now - cam.get("_last_alert_ts", 0) < cooldown:
            return

        cam["_last_alert_ts"] = now

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        jpeg_bytes = buf.tobytes() if ok and buf is not None else None
        score = cam.get("motion_score", 0.0)
        cam_name = cam.get("name", cam_id)

        active_zones = list(cam.get("active_zones", []))
        cam_notes = cam.get("notes", "")
        threading.Thread(
            target=_send_alert_email,
            args=(cam_id, cam_name, score, jpeg_bytes, active_zones,
                  cam_notes),
            daemon=True, name=f"alert-{cam_id}",
        ).start()

    # -- internal: connection ----------------------------------------

    def _connect(self, cam_id, cam):
        """Open a capture source. Returns the capture or None on failure."""
        try:
            cap = self._open_capture(cam)
            opened = cap.isOpened() if hasattr(cap, "isOpened") else True
            if not opened:
                self._release_cap(cap)
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
        if src == "hls":
            return _BrowserOnlyCapture()

        # RTSP (default)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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

        cv2.rectangle(f, (0, self.h // 2), (self.w, self.h),
                      (55, 58, 52), -1)
        cv2.line(f, (0, self.h // 2), (self.w, self.h // 2),
                 (70, 72, 68), 1)

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
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
        return ret, frame

    def release(self):
        self._cap.release()


class _HttpSnapshotCapture:
    """Polls an HTTP JPEG endpoint on each read().

    Tries ``requests`` first, falls back to ``curl`` subprocess for
    servers with restrictive TLS configurations.
    """

    def __init__(self, url, timeout=5):
        self.url = url
        self.timeout = timeout
        self._use_curl = False  # auto-detected on first failure

    def isOpened(self):
        # Try requests first
        try:
            r = requests.head(self.url, timeout=self.timeout)
            return r.status_code < 400
        except Exception:
            pass
        # Fallback: curl
        try:
            import subprocess
            result = subprocess.run(
                ["curl", "-sI", self.url],
                capture_output=True, timeout=self.timeout + 2)
            if b"200" in result.stdout[:50]:
                self._use_curl = True
                return True
        except Exception:
            pass
        return False

    def read(self):
        if self._use_curl:
            return self._read_curl()
        try:
            r = requests.get(self.url, timeout=self.timeout)
            r.raise_for_status()
            arr = np.frombuffer(r.content, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                return True, frame
        except Exception:
            # Switch to curl fallback for future calls
            self._use_curl = True
            return self._read_curl()
        return False, None

    def _read_curl(self):
        try:
            import subprocess
            result = subprocess.run(
                ["curl", "-s", self.url],
                capture_output=True, timeout=self.timeout + 2)
            if result.returncode == 0 and result.stdout:
                arr = np.frombuffer(result.stdout, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    return True, frame
        except Exception:
            pass
        return False, None

    def release(self):
        pass


class _BrowserOnlyCapture:
    """Sentinel for sources played directly in the browser (HLS/DASH)."""

    def isOpened(self):
        return True

    def read(self):
        return False, None

    def release(self):
        pass


# ─── Module singleton ──────────────────────────────────────────────────

_mgr = StreamManager()
atexit.register(_mgr.shutdown)
_auto_reconnect()


# ─── ONVIF Discovery ──────────────────────────────────────────────────

_onvif_available = None


def _check_onvif_deps():
    """Return True if onvif-zeep and wsdiscovery are installed."""
    global _onvif_available
    if _onvif_available is not None:
        return _onvif_available
    try:
        import onvif  # noqa: F401
        import wsdiscovery  # noqa: F401
        _onvif_available = True
    except ImportError:
        _onvif_available = False
    return _onvif_available


def _discover_onvif_devices(timeout=5):
    """Run WS-Discovery and return a list of discovered ONVIF device dicts."""
    from wsdiscovery.discovery import ThreadedWSDiscovery

    wsd = ThreadedWSDiscovery()
    wsd.start()
    try:
        services = wsd.searchServices(timeout=timeout)
    finally:
        wsd.stop()

    devices = []
    seen = set()
    for svc in services:
        scopes = [str(s) for s in (svc.getScopes() or [])]
        is_onvif = any("onvif" in s.lower() for s in scopes)
        if not is_onvif:
            continue
        for xaddr in svc.getXAddrs():
            if xaddr in seen:
                continue
            seen.add(xaddr)
            parsed = urlparse(xaddr)
            devices.append({
                "xaddr": xaddr,
                "ip": parsed.hostname or "",
                "port": parsed.port or 80,
                "scopes": scopes,
            })
    return devices


def _get_onvif_stream_uri(ip, port, username, password):
    """Connect to an ONVIF camera and return its first RTSP stream URI."""
    from onvif import ONVIFCamera

    cam = ONVIFCamera(ip, port, username, password)
    media = cam.create_media_service()
    profiles = media.GetProfiles()
    if not profiles:
        raise RuntimeError(f"No media profiles found on {ip}:{port}")

    results = []
    for prof in profiles:
        try:
            stream_setup = {
                "Stream": "RTP-Unicast",
                "Transport": {"Protocol": "RTSP"},
            }
            resp = media.GetStreamUri({
                "StreamSetup": stream_setup,
                "ProfileToken": prof.token,
            })
            uri = resp.Uri
            if username and "://" in uri:
                parsed = urlparse(uri)
                if not parsed.username:
                    uri = uri.replace(
                        "://", f"://{username}:{password}@", 1)
            results.append({
                "token": prof.token,
                "name": getattr(prof, "Name", prof.token),
                "uri": uri,
            })
        except Exception as exc:
            log.debug("Profile %s stream URI failed: %s", prof.token, exc)

    if not results:
        raise RuntimeError(f"Could not get stream URI from any profile "
                           f"on {ip}:{port}")
    return results[0]["uri"], results


_last_discovery = {"devices": [], "ts": 0}


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
            "camera_id", label="Camera ID", required=True,
            description="Unique identifier (e.g. lobby-1)",
        )
        inputs.str(
            "name", label="Name", required=True,
            description="Display name (e.g. Lobby Entrance)",
        )
        inputs.str(
            "url", label="Stream URL", required=True,
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
                types.Choice("hls", label="HLS / Web Stream (browser)"),
                types.Choice("http_snapshot",
                             label="HTTP Snapshot (periodic JPEG)"),
                types.Choice("video", label="Video File"),
                types.Choice("webcam", label="Local Webcam"),
                types.Choice("demo", label="Demo (synthetic)"),
            ]),
            default="rtsp", label="Source Type",
        )
        # Location
        inputs.view("loc_header", types.Notice(label="Location"))
        inputs.str("region", default="", label="Region",
                   description="State code or region (e.g. TX, CA)")
        inputs.str("locale", default="", label="Locale",
                   description="Site name (e.g. Store #1042 - Austin South)")
        inputs.enum(
            "room",
            values=types.Choices(choices=[
                types.Choice("", label="(none)"),
                types.Choice("lobby", label="Lobby"),
                types.Choice("sales-floor", label="Sales Floor"),
                types.Choice("stockroom", label="Stockroom"),
                types.Choice("office", label="Office"),
                types.Choice("bathroom", label="Bathroom"),
                types.Choice("parking-lot", label="Parking Lot"),
                types.Choice("entrance", label="Entrance"),
                types.Choice("loading-dock", label="Loading Dock"),
                types.Choice("hallway", label="Hallway"),
                types.Choice("server-room", label="Server Room"),
                types.Choice("other", label="Other..."),
            ]),
            default="", label="Room",
        )
        if ctx.params.get("room") == "other":
            inputs.str("room_custom", default="", label="Custom Room Name")
        inputs.str("position", default="", label="Position",
                   description="Mount point + orientation "
                               "(e.g. NW corner, facing entrance)")
        inputs.str("notes", default="", label="Notes",
                   description="Plain English instructions "
                               "(e.g. Alert manager if anyone enters "
                               "after 10pm)")
        return types.Property(inputs, view=types.View(
            label="Add Camera Stream"))

    def execute(self, ctx):
        cam_id = ctx.params["camera_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", cam_id):
            return {"added": False, "error": "camera_id must contain only "
                    "letters, digits, hyphens, and underscores"}
        room = ctx.params.get("room", "")
        if room == "other":
            room = ctx.params.get("room_custom", "")
        added = _mgr.add(
            cam_id, ctx.params["name"], ctx.params["url"],
            ctx.params.get("source_type", "rtsp"),
            region=ctx.params.get("region", ""),
            locale=ctx.params.get("locale", ""),
            room=room,
            position=ctx.params.get("position", ""),
            notes=ctx.params.get("notes", ""),
        )
        if added:
            _save_config()
        return {"added": added}

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.bool("added", label="Camera added")
        return types.Property(outputs)


class RemoveCamera(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="remove_camera", label="Remove Camera Stream", dynamic=True,
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
                label="Camera", required=True,
            )
        else:
            inputs.view("notice",
                        types.Notice(label="No cameras are connected"))
        return types.Property(inputs, view=types.View(
            label="Remove Camera"))

    def execute(self, ctx):
        cam_id = ctx.params.get("camera_id")
        removed = _mgr.remove(cam_id) if cam_id else False
        if removed:
            _save_config()
        return {"removed": removed}


class SnapshotCamera(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="snapshot_camera", label="Save Camera Snapshot",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        streaming = [
            c for c in _mgr.list_cameras()
            if c["status"] == "streaming" and c["source_type"] != "hls"
        ]
        if streaming:
            inputs.enum(
                "camera_id",
                values=types.Choices(choices=[
                    types.Choice(c["id"], label=c["name"])
                    for c in streaming
                ]),
                label="Camera", required=True,
            )
        else:
            inputs.view("notice",
                        types.Notice(label="No cameras currently streaming"))
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
            name="snapshot_all_cameras", label="Snapshot All Cameras",
        )

    def execute(self, ctx):
        if not ctx.dataset:
            return {"saved": 0}
        count = 0
        for cam in _mgr.list_cameras():
            if cam["status"] == "streaming" and cam["source_type"] != "hls":
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
            name="refresh_streams", label="Refresh Camera Streams",
        )

    def execute(self, ctx):
        cameras = _mgr.list_cameras()
        current_ids = {cam["id"] for cam in cameras}

        prev_ids = set(ctx.panel.state.get("_camera_ids") or [])
        state_patch = {"_cameras": cameras, "_camera_ids": list(current_ids)}
        for old_id in prev_ids - current_ids:
            state_patch[f"frame_{old_id}"] = None

        for cam in cameras:
            b64 = _mgr.frame_b64(cam["id"])
            if b64:
                state_patch[f"frame_{cam['id']}"] = (
                    f"data:image/jpeg;base64,{b64}")
            else:
                state_patch[f"frame_{cam['id']}"] = None

        # Build composite grid image
        grid_b64 = _mgr.grid_frame_b64()
        if grid_b64:
            state_patch["_grid_frame"] = (
                f"data:image/jpeg;base64,{grid_b64}")
        else:
            state_patch["_grid_frame"] = None

        ctx.panel.state.set(state_patch)
        return {}


class DiscoverCameras(foo.Operator):
    """Scan the local network for ONVIF cameras via WS-Discovery."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="discover_cameras", label="Discover ONVIF Cameras",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        if not _check_onvif_deps():
            inputs.view("missing_deps", types.Notice(
                label="ONVIF discovery requires additional packages. "
                      "Run: pip install onvif-zeep WSDiscovery"))
            return types.Property(inputs, view=types.View(
                label="Discover ONVIF Cameras"))

        devices = _last_discovery["devices"]
        age = time.time() - _last_discovery["ts"]

        should_scan = ctx.params.get("scan", False)
        if should_scan or not devices:
            try:
                devices = _discover_onvif_devices(timeout=5)
                _last_discovery["devices"] = devices
                _last_discovery["ts"] = time.time()
            except Exception as exc:
                inputs.view("scan_error", types.Notice(
                    label=f"Discovery failed: {exc}"))
                return types.Property(inputs, view=types.View(
                    label="Discover ONVIF Cameras"))

        inputs.bool("scan", default=False, label="Re-scan network",
                    description="Check this box to re-run WS-Discovery")

        if not devices:
            inputs.view("no_devices", types.Notice(
                label="No ONVIF cameras found on the local network."))
            return types.Property(inputs, view=types.View(
                label="Discover ONVIF Cameras"))

        if age > 0:
            inputs.view("scan_info", types.Notice(
                label=f"Found {len(devices)} ONVIF device(s) "
                      f"({int(age)}s ago)"))

        inputs.enum(
            "device_idx",
            values=types.Choices(choices=[
                types.Choice(str(i), label=f"{d['ip']}:{d['port']}",
                             description=d["xaddr"])
                for i, d in enumerate(devices)
            ]),
            label="Camera", required=True,
            description="Select a discovered ONVIF device",
        )
        inputs.str("username", label="Username", default="admin",
                   description="ONVIF username")
        inputs.str("password", label="Password", default="",
                   description="ONVIF password")
        inputs.str("camera_id", label="Camera ID",
                   description="Unique ID (auto-generated if blank)")
        inputs.str("name", label="Display Name",
                   description="Friendly name (defaults to IP address)")
        # Location
        inputs.view("loc_header", types.Notice(label="Location"))
        inputs.str("region", default="", label="Region",
                   description="State code or region (e.g. TX, CA)")
        inputs.str("locale_name", default="", label="Locale",
                   description="Site name (e.g. Store #1042)")
        inputs.str("room", default="", label="Room",
                   description="e.g. lobby, parking-lot, entrance")
        inputs.str("position", default="", label="Position",
                   description="Mount point + orientation")
        inputs.str("notes", default="", label="Notes",
                   description="Plain English instructions")

        return types.Property(inputs, view=types.View(
            label="Discover ONVIF Cameras"))

    def execute(self, ctx):
        if not _check_onvif_deps():
            return {"added": False, "error": "ONVIF deps not installed"}

        idx_str = ctx.params.get("device_idx")
        if idx_str is None:
            return {"added": False, "error": "No device selected"}

        devices = _last_discovery["devices"]
        idx = int(idx_str)
        if idx < 0 or idx >= len(devices):
            return {"added": False, "error": "Invalid device index"}

        dev = devices[idx]
        username = ctx.params.get("username", "admin")
        password = ctx.params.get("password", "")

        try:
            rtsp_uri, profiles = _get_onvif_stream_uri(
                dev["ip"], dev["port"], username, password)
        except Exception as exc:
            return {"added": False, "error": str(exc)}

        cam_id = (ctx.params.get("camera_id") or "").strip()
        if not cam_id:
            cam_id = f"onvif-{dev['ip'].replace('.', '-')}"
        if not re.fullmatch(r"[A-Za-z0-9_-]+", cam_id):
            return {"added": False, "error": "camera_id must contain only "
                    "letters, digits, hyphens, and underscores"}

        name = (ctx.params.get("name") or "").strip() or f"ONVIF {dev['ip']}"

        added = _mgr.add(
            cam_id, name, rtsp_uri, source_type="rtsp",
            region=ctx.params.get("region", ""),
            locale=ctx.params.get("locale_name", ""),
            room=ctx.params.get("room", ""),
            position=ctx.params.get("position", ""),
            notes=ctx.params.get("notes", ""),
        )
        if added:
            _save_config()
        return {
            "added": added, "camera_id": cam_id, "rtsp_uri": rtsp_uri,
            "profiles": [p["name"] for p in profiles],
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.bool("added", label="Camera added")
        outputs.str("camera_id", label="Camera ID")
        outputs.str("rtsp_uri", label="RTSP URI")
        return types.Property(outputs)


# ─── NVR Operators ─────────────────────────────────────────────────────


class ConfigureMotion(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="configure_motion",
            label="Configure Motion Detection",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        cams = _mgr.list_cameras()
        if not cams:
            inputs.view("notice",
                        types.Notice(label="No cameras connected"))
            return types.Property(inputs, view=types.View(
                label="Configure Motion"))

        inputs.enum(
            "camera_id",
            values=types.Choices(choices=[
                types.Choice(c["id"], label=f"{c['name']} ({c['id']})")
                for c in cams
            ]),
            label="Camera", required=True,
        )
        inputs.bool("enabled", default=True,
                    label="Enable Motion Detection")
        inputs.str("threshold", default="25",
                   label="Pixel Diff Threshold (1-100)",
                   description="Lower = more sensitive")
        inputs.str("min_area", default="500",
                   label="Minimum Contour Area (pixels)",
                   description="Ignore motion regions smaller than this")
        return types.Property(inputs, view=types.View(
            label="Configure Motion Detection"))

    def execute(self, ctx):
        cam_id = ctx.params.get("camera_id")
        if not cam_id:
            return {"configured": False}
        try:
            threshold = int(ctx.params.get("threshold", 25))
            min_area = int(ctx.params.get("min_area", 500))
        except (ValueError, TypeError):
            return {"configured": False, "error": "Invalid numeric values"}
        ok = _mgr.configure_motion(
            cam_id,
            enabled=ctx.params.get("enabled", True),
            threshold=max(1, min(threshold, 100)),
            min_area=max(0, min_area),
        )
        if ok:
            _save_config()
        return {"configured": ok}


class ToggleRecording(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="toggle_recording",
            label="Toggle Recording",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        cams = [c for c in _mgr.list_cameras()
                if c["source_type"] != "hls"]
        if not cams:
            inputs.view("notice",
                        types.Notice(label="No recordable cameras connected"))
            return types.Property(inputs, view=types.View(
                label="Toggle Recording"))

        inputs.enum(
            "camera_id",
            values=types.Choices(choices=[
                types.Choice(
                    c["id"],
                    label=(f"{c['name']} ({c['id']}) — "
                           f"{'REC' if c.get('recording_active') else c.get('recording_mode', 'off')}"),
                )
                for c in cams
            ]),
            label="Camera", required=True,
        )
        inputs.enum(
            "mode",
            values=types.Choices(choices=[
                types.Choice("off", label="Off"),
                types.Choice("continuous", label="Continuous"),
                types.Choice("motion", label="Motion-Triggered"),
            ]),
            default="motion", label="Recording Mode",
        )
        return types.Property(inputs, view=types.View(
            label="Toggle Recording"))

    def execute(self, ctx):
        cam_id = ctx.params.get("camera_id")
        mode = ctx.params.get("mode", "off")
        if not cam_id:
            return {"configured": False}
        # Start recording indexer when enabling recording
        if mode != "off" and ctx.dataset:
            _mgr.enable_recording_index(ctx.dataset.name)
        ok = _mgr.configure_recording(cam_id, mode=mode)
        if ok:
            _save_config()
        return {"configured": ok, "mode": mode}


class ConfigureRecording(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="configure_recording",
            label="Configure Recording Settings",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        cams = [c for c in _mgr.list_cameras()
                if c["source_type"] != "hls"]
        if not cams:
            inputs.view("notice",
                        types.Notice(label="No recordable cameras connected"))
            return types.Property(inputs, view=types.View(
                label="Configure Recording"))

        inputs.enum(
            "camera_id",
            values=types.Choices(choices=[
                types.Choice(c["id"], label=f"{c['name']} ({c['id']})")
                for c in cams
            ]),
            label="Camera", required=True,
        )
        inputs.str("output_dir", default="securvision_recordings",
                   label="Output Directory")
        inputs.str("segment_seconds", default="300",
                   label="Segment Duration (seconds)")
        inputs.str("preroll_seconds", default="5",
                   label="Pre-roll Buffer (seconds)")
        inputs.str("postroll_seconds", default="10",
                   label="Post-roll Duration (seconds)")
        inputs.str("fps", default="10", label="Recording FPS")
        return types.Property(inputs, view=types.View(
            label="Configure Recording"))

    def execute(self, ctx):
        cam_id = ctx.params.get("camera_id")
        if not cam_id:
            return {"configured": False}
        try:
            ok = _mgr.configure_recording(
                cam_id,
                output_dir=ctx.params.get("output_dir",
                                          "securvision_recordings"),
                segment_seconds=int(
                    ctx.params.get("segment_seconds", 300)),
                preroll_seconds=int(
                    ctx.params.get("preroll_seconds", 5)),
                postroll_seconds=int(
                    ctx.params.get("postroll_seconds", 10)),
                fps=int(ctx.params.get("fps", 10)),
            )
        except (ValueError, TypeError):
            return {"configured": False, "error": "Invalid numeric values"}
        if ok:
            _save_config()
        return {"configured": ok}


class ConfigureAlerts(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="configure_alerts",
            label="Configure Email Alerts",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.bool("enabled",
                    default=_alert_config.get("enabled", False),
                    label="Enable Email Alerts")
        inputs.str("smtp_host",
                   default=_alert_config.get("smtp_host", ""),
                   label="SMTP Host")
        inputs.str("smtp_port",
                   default=str(_alert_config.get("smtp_port", 587)),
                   label="SMTP Port")
        inputs.bool("smtp_tls",
                    default=_alert_config.get("smtp_tls", True),
                    label="Use TLS")
        inputs.str("sender",
                   default=_alert_config.get("sender", ""),
                   label="Sender Email")
        inputs.str("password", default="", label="SMTP Password",
                   description="App password or SMTP credentials")
        inputs.str("recipients",
                   default=", ".join(
                       _alert_config.get("recipients", [])),
                   label="Recipients (comma-separated)")
        inputs.str("cooldown_seconds",
                   default=str(
                       _alert_config.get("cooldown_seconds", 300)),
                   label="Cooldown (seconds between alerts per camera)")

        cams = _mgr.list_cameras()
        if cams:
            inputs.view("cam_header", types.Notice(
                label="Per-camera alert enable/disable:"))
            for c in cams:
                inputs.bool(
                    f"alert_{c['id']}",
                    default=c.get("alerts_enabled", True),
                    label=f"Alerts for {c['name']} ({c['id']})",
                )

        return types.Property(inputs, view=types.View(
            label="Configure Email Alerts"))

    def execute(self, ctx):
        _alert_config["enabled"] = ctx.params.get("enabled", False)
        _alert_config["smtp_host"] = ctx.params.get("smtp_host", "")
        try:
            _alert_config["smtp_port"] = int(
                ctx.params.get("smtp_port", 587))
        except (ValueError, TypeError):
            _alert_config["smtp_port"] = 587
        _alert_config["smtp_tls"] = ctx.params.get("smtp_tls", True)
        _alert_config["sender"] = ctx.params.get("sender", "")
        pw = ctx.params.get("password", "")
        if pw:
            _alert_config["password"] = pw
        recipients_str = ctx.params.get("recipients", "")
        _alert_config["recipients"] = [
            r.strip() for r in recipients_str.split(",") if r.strip()
        ]
        try:
            _alert_config["cooldown_seconds"] = int(
                ctx.params.get("cooldown_seconds", 300))
        except (ValueError, TypeError):
            _alert_config["cooldown_seconds"] = 300

        for cam in _mgr.list_cameras():
            key = f"alert_{cam['id']}"
            if key in ctx.params:
                _mgr.set_alerts_enabled(cam["id"], ctx.params[key])

        _save_config()
        return {"configured": True}


# ─── Detection & Search Operators ──────────────────────────────────────


class ConfigureDetection(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="configure_detection",
            label="Configure Object Detection",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        if not _check_detection_deps():
            inputs.view("notice", types.Notice(
                label="Object detection requires fiftyone[zoo] and torch. "
                      "Run: pip install 'fiftyone[zoo]' torch ultralytics"))
            return types.Property(inputs, view=types.View(
                label="Configure Detection"))

        inputs.str("model_name",
                   default=_detection_config["model_name"],
                   label="Model Name",
                   description="FiftyOne zoo model (e.g. yolov8s-coco-torch) "
                               "or path to custom model")
        inputs.str("confidence_threshold",
                   default=str(_detection_config["confidence_threshold"]),
                   label="Confidence Threshold (0-1)")
        inputs.str("label_field",
                   default=_detection_config["label_field"],
                   label="Detection Field Name")
        inputs.bool("auto_detect_snapshots",
                    default=_detection_config["auto_detect_snapshots"],
                    label="Auto-detect on new snapshots")
        inputs.bool("overlay_enabled",
                    default=_detection_config.get("overlay_enabled", False),
                    label="Show live detection overlay on feeds")
        inputs.str("overlay_interval",
                   default=str(_detection_config.get("overlay_interval", 5)),
                   label="Overlay interval (run every N frames)",
                   description="Higher = less CPU. 5 = ~2fps detection "
                               "at 10fps capture")
        inputs.enum(
            "overlay_mode",
            values=types.Choices(choices=[
                types.Choice("box",
                             label="Bounding Boxes (fast)"),
                types.Choice("segmentation",
                             label="Segmentation Masks (organic shapes, slower)"),
            ]),
            default=_detection_config.get("overlay_mode", "box"),
            label="Overlay Style",
            description="Box = rectangles. Segmentation = pixel-accurate "
                        "outlines tracing object shapes (2x slower).",
        )
        inputs.str("seg_opacity",
                   default=str(_detection_config.get("seg_opacity", 0.4)),
                   label="Segmentation mask opacity (0.0-1.0)",
                   description="How transparent the colored overlay is. "
                               "0 = invisible, 1 = solid color.")
        return types.Property(inputs, view=types.View(
            label="Configure Object Detection"))

    def execute(self, ctx):
        old_name = _detection_config["model_name"]
        _detection_config["model_name"] = ctx.params.get(
            "model_name", old_name)
        try:
            _detection_config["confidence_threshold"] = float(
                ctx.params.get("confidence_threshold", 0.25))
        except (ValueError, TypeError):
            pass
        _detection_config["label_field"] = ctx.params.get(
            "label_field", "detections")
        _detection_config["auto_detect_snapshots"] = ctx.params.get(
            "auto_detect_snapshots", False)
        _detection_config["overlay_enabled"] = ctx.params.get(
            "overlay_enabled", False)
        try:
            _detection_config["overlay_interval"] = max(1, int(
                ctx.params.get("overlay_interval", 5)))
        except (ValueError, TypeError):
            pass
        _detection_config["overlay_mode"] = ctx.params.get(
            "overlay_mode", "box")
        try:
            _detection_config["seg_opacity"] = max(0.0, min(1.0, float(
                ctx.params.get("seg_opacity", 0.4))))
        except (ValueError, TypeError):
            pass
        # Force model reload if name changed
        global _detection_model, _detection_model_name
        if _detection_config["model_name"] != old_name:
            with _model_lock:
                _detection_model = None
                _detection_model_name = None
        _save_config()
        return {"configured": True}


class AnalyzeSamples(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="analyze_samples",
            label="Analyze Samples (Object Detection)",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        if not _check_detection_deps():
            inputs.view("notice", types.Notice(
                label="Object detection requires fiftyone[zoo] and torch. "
                      "Run: pip install 'fiftyone[zoo]' torch ultralytics"))
            return types.Property(inputs, view=types.View(
                label="Analyze Samples"))

        inputs.enum(
            "target",
            values=types.Choices(choices=[
                types.Choice("current_view", label="Current View"),
                types.Choice("all", label="All Samples"),
            ]),
            default="current_view", label="Apply detection to",
        )
        inputs.str("model_name",
                   default=_detection_config["model_name"],
                   label="Model Name")
        inputs.str("label_field",
                   default=_detection_config["label_field"],
                   label="Detection Field Name")
        inputs.str("confidence_threshold",
                   default=str(_detection_config["confidence_threshold"]),
                   label="Confidence Threshold")
        return types.Property(inputs, view=types.View(
            label="Analyze Samples"))

    def execute(self, ctx):
        if not ctx.dataset:
            return {"analyzed": 0, "error": "No dataset loaded"}
        target = ctx.params.get("target", "current_view")
        model_name = ctx.params.get("model_name",
                                    _detection_config["model_name"])
        label_field = ctx.params.get("label_field",
                                     _detection_config["label_field"])
        try:
            conf = float(ctx.params.get("confidence_threshold", 0.25))
        except (ValueError, TypeError):
            conf = 0.25

        try:
            model = _get_detection_model(model_name)
        except Exception as exc:
            return {"analyzed": 0, "error": str(exc)}

        if target == "current_view" and ctx.view is not None:
            view = ctx.view
        else:
            view = ctx.dataset.view()

        count = len(view)
        view.apply_model(model, label_field=label_field)

        # Filter low-confidence detections and compute counts
        for sample in view:
            dets = sample.get_field(label_field)
            if dets and hasattr(dets, "detections"):
                if conf > 0:
                    dets.detections = [
                        d for d in dets.detections
                        if (d.confidence or 0) >= conf
                    ]
                counts = {}
                for d in dets.detections:
                    counts[d.label] = counts.get(d.label, 0) + 1
                sample["detection_counts"] = counts
                sample.save()

        ctx.trigger("reload_dataset")
        return {"analyzed": count, "label_field": label_field}


class SearchFootage(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="search_footage",
            label="Search Footage",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        if not ctx.dataset:
            inputs.view("notice",
                        types.Notice(label="No dataset loaded"))
            return types.Property(inputs, view=types.View(
                label="Search Footage"))

        # Camera selector from dataset
        try:
            cam_ids = ctx.dataset.distinct("camera_id") or []
        except Exception:
            cam_ids = []
        cam_choices = [types.Choice("all", label="All Cameras")]
        for cid in cam_ids:
            if cid:
                cam_choices.append(types.Choice(cid, label=str(cid)))
        inputs.enum("camera_id",
                    values=types.Choices(choices=cam_choices),
                    default="all", label="Camera")
        inputs.str("region_filter", default="", label="Region",
                   description="Filter by state/region (e.g. TX)")
        inputs.str("locale_filter", default="", label="Locale",
                   description="Filter by site name")
        inputs.str("room_filter", default="", label="Room",
                   description="Filter by room (e.g. parking-lot)")

        now = datetime.datetime.now()
        day_ago = now - datetime.timedelta(days=1)
        inputs.str("start_date",
                   default=day_ago.strftime("%Y-%m-%d %H:%M:%S"),
                   label="Start Date/Time",
                   description="YYYY-MM-DD HH:MM:SS")
        inputs.str("end_date",
                   default=now.strftime("%Y-%m-%d %H:%M:%S"),
                   label="End Date/Time",
                   description="YYYY-MM-DD HH:MM:SS")
        inputs.enum(
            "sample_type",
            values=types.Choices(choices=[
                types.Choice("all", label="All"),
                types.Choice("snapshot", label="Snapshots"),
                types.Choice("recording", label="Recordings"),
            ]),
            default="all", label="Sample Type",
        )
        inputs.str("detection_label", default="",
                   label="Detection Label",
                   description="e.g. person, car, dog (empty for any)")
        inputs.str("min_confidence", default="0.0",
                   label="Min Detection Confidence")
        inputs.bool("motion_only", default=False,
                    label="Motion events only")
        inputs.str("zone_filter", default="", label="Zone",
                   description="Filter by zone name (e.g. entrance, center)")
        inputs.str("min_persons", default="0",
                   label="Min Person Count",
                   description="Only show samples with at least N persons")
        return types.Property(inputs, view=types.View(
            label="Search Footage"))

    def execute(self, ctx):
        if not ctx.dataset:
            return {"matched": 0}
        from fiftyone import ViewField as F

        view = ctx.dataset.view()
        camera_id = ctx.params.get("camera_id", "all")
        if camera_id != "all":
            view = view.match(F("camera_id") == camera_id)

        # Location filters
        for loc_param, loc_field in [("region_filter", "region"),
                                     ("locale_filter", "locale"),
                                     ("room_filter", "room")]:
            val = (ctx.params.get(loc_param) or "").strip()
            if val:
                view = view.match(F(loc_field) == val)

        # Parse dates
        for field, param in [("captured_at", "start_date"),
                             ("captured_at", "end_date")]:
            val = ctx.params.get(param, "")
            if not val:
                continue
            try:
                dt = datetime.datetime.fromisoformat(val)
            except ValueError:
                try:
                    dt = datetime.datetime.strptime(
                        val, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
            if param == "start_date":
                view = view.match(F("captured_at") >= dt)
            else:
                view = view.match(F("captured_at") <= dt)

        sample_type = ctx.params.get("sample_type", "all")
        if sample_type == "snapshot":
            view = view.match_tags("snapshot")
        elif sample_type == "recording":
            view = view.match_tags("recording")

        if ctx.params.get("motion_only", False):
            view = view.match(F("motion_detected") == True)  # noqa: E712

        zone_val = (ctx.params.get("zone_filter") or "").strip()
        if zone_val:
            view = view.match(F("active_zones").contains(zone_val))

        try:
            min_p = int(ctx.params.get("min_persons", 0))
        except (ValueError, TypeError):
            min_p = 0
        if min_p > 0:
            view = view.match(
                F("detection_counts.person") >= min_p)

        det_label = (ctx.params.get("detection_label") or "").strip()
        try:
            min_conf = float(ctx.params.get("min_confidence", 0))
        except (ValueError, TypeError):
            min_conf = 0.0
        label_field = _detection_config["label_field"]

        if det_label:
            expr = F("label") == det_label
            if min_conf > 0:
                expr = expr & (F("confidence") >= min_conf)
            view = view.filter_labels(label_field, expr)
        elif min_conf > 0:
            view = view.filter_labels(
                label_field, F("confidence") >= min_conf)

        count = len(view)
        # Apply the filtered view to the FiftyOne App
        try:
            ctx.trigger(
                "@voxel51/operators/set_view",
                params={"view": view._serialize()},
            )
        except Exception:
            # Fallback: reload to show updated dataset
            ctx.trigger("reload_dataset")
        return {"matched": count}

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("matched", label="Samples matched")
        return types.Property(outputs)


class BrowseTimeline(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="browse_timeline",
            label="Browse Timeline",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        if not ctx.dataset:
            inputs.view("notice",
                        types.Notice(label="No dataset loaded"))
            return types.Property(inputs, view=types.View(
                label="Browse Timeline"))

        try:
            cam_ids = ctx.dataset.distinct("camera_id") or []
        except Exception:
            cam_ids = []
        cam_choices = [types.Choice("all", label="All Cameras")]
        for cid in cam_ids:
            if cid:
                cam_choices.append(types.Choice(cid, label=str(cid)))
        inputs.enum("camera_id",
                    values=types.Choices(choices=cam_choices),
                    default="all", label="Camera")

        now = datetime.datetime.now()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        inputs.str("start_date",
                   default=today.strftime("%Y-%m-%d %H:%M:%S"),
                   label="Start Date/Time")
        inputs.str("end_date",
                   default=now.strftime("%Y-%m-%d %H:%M:%S"),
                   label="End Date/Time")
        return types.Property(inputs, view=types.View(
            label="Browse Timeline"))

    def execute(self, ctx):
        if not ctx.dataset:
            return {}
        from fiftyone import ViewField as F

        camera_id = ctx.params.get("camera_id", "all")
        start_str = ctx.params.get("start_date", "")
        end_str = ctx.params.get("end_date", "")

        try:
            start_dt = datetime.datetime.fromisoformat(start_str)
        except (ValueError, TypeError):
            start_dt = datetime.datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0)
        try:
            end_dt = datetime.datetime.fromisoformat(end_str)
        except (ValueError, TypeError):
            end_dt = datetime.datetime.now()

        view = ctx.dataset.match(
            (F("captured_at") >= start_dt) & (F("captured_at") <= end_dt))
        if camera_id != "all":
            view = view.match(F("camera_id") == camera_id)

        md = _render_timeline_markdown(view, start_dt, end_dt)
        ctx.panel.state.set({
            "_timeline_md": md,
            "_timeline_start": start_str,
            "_timeline_end": end_str,
        })
        return {}


def _render_timeline_markdown(view, start_dt, end_dt):
    """Build a markdown timeline from dataset samples."""
    header = (
        f"### Timeline: {start_dt.strftime('%Y-%m-%d %H:%M')} to "
        f"{end_dt.strftime('%Y-%m-%d %H:%M')}\n\n"
    )

    events_by_cam = {}
    count = 0
    max_events = 200
    for sample in view:
        count += 1
        if count > max_events:
            break
        cam_id = sample.get_field("camera_id") or "unknown"
        cam_name = sample.get_field("camera_name") or cam_id
        loc_parts = [p for p in (sample.get_field("region") or "",
                                 sample.get_field("locale") or "",
                                 sample.get_field("room") or "") if p]
        cam_loc = " > ".join(loc_parts)
        key = (cam_id, cam_name, cam_loc)
        if key not in events_by_cam:
            events_by_cam[key] = []

        ts = sample.get_field("captured_at")
        ts_str = ts.strftime("%H:%M:%S") if ts else "?"
        tags = sample.tags or []
        sample_type = sample.get_field("sample_type") or (
            "recording" if "recording" in tags else "snapshot")

        # Detection summary
        det_summary = ""
        label_field = _detection_config["label_field"]
        dets = sample.get_field(label_field)
        if dets and hasattr(dets, "detections") and dets.detections:
            labels = {}
            for d in dets.detections:
                labels[d.label] = labels.get(d.label, 0) + 1
            det_summary = ", ".join(
                f"{lbl} x{cnt}" for lbl, cnt in sorted(labels.items()))

        # Build row
        if sample_type == "recording":
            started = sample.get_field("started_at")
            ended = sample.get_field("ended_at")
            dur = sample.get_field("duration_seconds") or 0
            mode = sample.get_field("recording_mode") or "?"
            t_range = (
                f"{started.strftime('%H:%M:%S') if started else '?'}"
                f" - {ended.strftime('%H:%M:%S') if ended else '?'}")
            dur_str = f"{int(dur // 60)}m {int(dur % 60)}s"
            details = f"{mode}, {dur_str}"
            if det_summary:
                details += f" ({det_summary})"
            events_by_cam[key].append(
                (started or ts, f"| {t_range} | REC | {details} |"))
        else:
            motion = sample.get_field("motion_detected")
            m_score = sample.get_field("motion_score") or 0
            details = ""
            if motion:
                details = f"MOTION ({m_score:.1%})"
            if det_summary:
                details += f" {det_summary}" if details else det_summary
            if not details:
                details = "snapshot"
            events_by_cam[key].append(
                (ts, f"| {ts_str} | SNAP | {details} |"))

    if not events_by_cam:
        return header + "*No events found in this time range.*\n"

    body = header
    for (cam_id, cam_name, cam_loc), events in sorted(events_by_cam.items()):
        events.sort(key=lambda e: e[0] if e[0] else datetime.datetime.min)
        loc_suffix = f" ({cam_loc})" if cam_loc else ""
        body += f"#### {cam_name}{loc_suffix} (`{cam_id}`)\n\n"
        body += "| Time | Type | Details |\n|------|------|------|\n"
        for _, row in events:
            body += row + "\n"
        body += f"\n*{len(events)} event(s)*\n\n---\n\n"

    if count > max_events:
        body += (f"\n*Showing first {max_events} events. "
                 f"Narrow the date range for more detail.*\n")

    return body


class ConfigureZones(foo.Operator):
    """Configure 3x3 detection zone grid for a camera."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="configure_zones", label="Configure Zones", dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        cams = _mgr.list_cameras()
        if not cams:
            inputs.view("notice",
                        types.Notice(label="No cameras connected"))
            return types.Property(inputs, view=types.View(
                label="Configure Zones"))

        inputs.enum(
            "camera_id",
            values=types.Choices(choices=[
                types.Choice(c["id"], label=f"{c['name']} ({c['id']})")
                for c in cams
            ]),
            label="Camera", required=True,
        )

        inputs.view("grid_help", types.Notice(
            label="Zone Grid: toggle cells on/off. "
                  "Only enabled cells trigger motion/alerts."))
        inputs.str("grid_diagram", default=(
            "```\n"
            "+----------+----------+----------+\n"
            "| TL (0,0) | TC (1,0) | TR (2,0) |\n"
            "+----------+----------+----------+\n"
            "| ML (0,1) | MC (1,1) | MR (2,1) |\n"
            "+----------+----------+----------+\n"
            "| BL (0,2) | BC (1,2) | BR (2,2) |\n"
            "+----------+----------+----------+\n"
            "```"
        ), view=types.MarkdownView())

        # Get current zone config for selected camera
        selected = ctx.params.get("camera_id")
        cur_zones = [[True]*3, [True]*3, [True]*3]
        cur_names = [
            ["top-left", "top-center", "top-right"],
            ["mid-left", "center", "mid-right"],
            ["bottom-left", "bottom-center", "bottom-right"],
        ]
        if selected:
            with _mgr._lock:
                cam = _mgr._cams.get(selected)
                if cam:
                    cur_zones = cam.get("zones", cur_zones)
                    cur_names = cam.get("zone_names", cur_names)

        labels = [
            ["Top-Left", "Top-Center", "Top-Right"],
            ["Mid-Left", "Center", "Mid-Right"],
            ["Bottom-Left", "Bottom-Center", "Bottom-Right"],
        ]
        for row in range(3):
            for col in range(3):
                key = f"zone_{row}_{col}"
                name_key = f"name_{row}_{col}"
                inputs.bool(key, default=cur_zones[row][col],
                            label=f"{labels[row][col]} — enabled")
                if ctx.params.get(key, cur_zones[row][col]):
                    inputs.str(name_key, default=cur_names[row][col],
                               label=f"{labels[row][col]} — name")

        # ── Custom named zones (rectangles) ──
        cur_custom = []
        if selected:
            with _mgr._lock:
                cam = _mgr._cams.get(selected)
                if cam:
                    cur_custom = cam.get("custom_zones", [])

        inputs.view("custom_header", types.Notice(
            label="Custom Zones: define named rectangles using "
                  "normalized coordinates (0.0-1.0). "
                  "Custom zones override the grid when set."))
        if cur_custom:
            summary = " | ".join(
                f"{z['name']} ({z['x']:.1f},{z['y']:.1f} "
                f"{z['w']:.1f}x{z['h']:.1f})"
                for z in cur_custom)
            inputs.str("cur_zones_display",
                       default=f"Current: {summary}",
                       view=types.MarkdownView())
        inputs.str("new_zone_name", default="", label="Add Zone — Name",
                   description="e.g. entrance, register, driveway")
        inputs.str("new_zone_x", default="0.0",
                   label="X (left edge, 0.0-1.0)")
        inputs.str("new_zone_y", default="0.0",
                   label="Y (top edge, 0.0-1.0)")
        inputs.str("new_zone_w", default="0.5",
                   label="Width (0.0-1.0)")
        inputs.str("new_zone_h", default="0.5",
                   label="Height (0.0-1.0)")
        if cur_custom:
            inputs.enum(
                "remove_zone",
                values=types.Choices(choices=[
                    types.Choice("", label="(none)")] + [
                    types.Choice(z["name"], label=f"Remove: {z['name']}")
                    for z in cur_custom
                ]),
                default="", label="Remove Zone",
            )
        inputs.bool("clear_custom", default=False,
                    label="Clear all custom zones (revert to grid)")

        return types.Property(inputs, view=types.View(
            label="Configure Zones"))

    def execute(self, ctx):
        cam_id = ctx.params.get("camera_id")
        if not cam_id:
            return {"configured": False}

        default_names = [
            ["top-left", "top-center", "top-right"],
            ["mid-left", "center", "mid-right"],
            ["bottom-left", "bottom-center", "bottom-right"],
        ]

        zones = []
        zone_names = []
        for row in range(3):
            zone_row = []
            name_row = []
            for col in range(3):
                enabled = ctx.params.get(f"zone_{row}_{col}", True)
                name = (ctx.params.get(f"name_{row}_{col}") or "").strip()
                if not name:
                    name = default_names[row][col]
                zone_row.append(enabled)
                name_row.append(name)
            zones.append(zone_row)
            zone_names.append(name_row)

        with _mgr._lock:
            cam = _mgr._cams.get(cam_id)
            if not cam:
                return {"configured": False}
            cam["zones"] = zones
            cam["zone_names"] = zone_names

            # Handle custom zones
            custom = list(cam.get("custom_zones", []))

            if ctx.params.get("clear_custom", False):
                custom = []
            else:
                # Remove zone
                remove = (ctx.params.get("remove_zone") or "").strip()
                if remove:
                    custom = [z for z in custom if z["name"] != remove]
                # Add new zone
                new_name = (ctx.params.get("new_zone_name") or "").strip()
                if new_name:
                    try:
                        zx = max(0.0, min(float(
                            ctx.params.get("new_zone_x", 0)), 1.0))
                        zy = max(0.0, min(float(
                            ctx.params.get("new_zone_y", 0)), 1.0))
                        zw = max(0.01, min(float(
                            ctx.params.get("new_zone_w", 0.5)),
                            1.0 - zx))
                        zh = max(0.01, min(float(
                            ctx.params.get("new_zone_h", 0.5)),
                            1.0 - zy))
                        custom.append({
                            "name": new_name,
                            "x": zx, "y": zy, "w": zw, "h": zh,
                        })
                    except (ValueError, TypeError):
                        pass

            cam["custom_zones"] = custom
            cam["_zone_mask"] = None  # invalidate cached mask

        _save_config()
        return {"configured": True}


class ImportZonesFromSample(foo.Operator):
    """Import zone definitions from bounding box annotations on a snapshot."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="import_zones_from_sample",
            label="Import Zones from Annotations",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        cams = _mgr.list_cameras()
        if not cams:
            inputs.view("notice",
                        types.Notice(label="No cameras connected"))
            return types.Property(inputs, view=types.View(
                label="Import Zones"))

        inputs.enum(
            "camera_id",
            values=types.Choices(choices=[
                types.Choice(c["id"], label=f"{c['name']} ({c['id']})")
                for c in cams
            ]),
            label="Target Camera", required=True,
        )
        inputs.str("label_field", default="zones",
                   label="Annotation Field",
                   description="Field containing bounding box annotations "
                               "drawn in the FiftyOne App (default: zones)")
        inputs.view("instructions", types.Notice(
            label="How to use: 1) Take a snapshot of the camera. "
                  "2) In the FiftyOne grid, click the snapshot and draw "
                  "bounding boxes using the annotation tools. Label each "
                  "box with the zone name (e.g. 'entrance', 'register'). "
                  "3) Select the annotated sample and run this operator."))
        return types.Property(inputs, view=types.View(
            label="Import Zones from Annotations"))

    def execute(self, ctx):
        cam_id = ctx.params.get("camera_id")
        label_field = ctx.params.get("label_field", "zones")
        if not cam_id or not ctx.dataset:
            return {"imported": 0}

        # Find samples for this camera that have annotations
        from fiftyone import ViewField as F
        view = ctx.dataset.match(F("camera_id") == cam_id)

        custom_zones = []
        for sample in view:
            dets = sample.get_field(label_field)
            if not dets or not hasattr(dets, "detections"):
                continue
            for det in dets.detections:
                bb = det.bounding_box  # [x, y, w, h] normalized
                if bb and len(bb) == 4:
                    custom_zones.append({
                        "name": det.label or "zone",
                        "x": bb[0], "y": bb[1],
                        "w": bb[2], "h": bb[3],
                    })
            break  # Use first annotated sample only

        if not custom_zones:
            return {"imported": 0,
                    "error": "No bounding box annotations found"}

        with _mgr._lock:
            cam = _mgr._cams.get(cam_id)
            if not cam:
                return {"imported": 0}
            cam["custom_zones"] = custom_zones
            cam["_zone_mask"] = None

        _save_config()
        return {"imported": len(custom_zones),
                "zones": [z["name"] for z in custom_zones]}

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("imported", label="Zones imported")
        return types.Property(outputs)


class ManageCameras(foo.Operator):
    """Edit location and display name on an existing camera."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="manage_cameras", label="Edit Camera", dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        cams = _mgr.list_cameras()
        if not cams:
            inputs.view("notice",
                        types.Notice(label="No cameras connected"))
            return types.Property(inputs, view=types.View(
                label="Edit Camera"))

        inputs.enum(
            "camera_id",
            values=types.Choices(choices=[
                types.Choice(c["id"], label=f"{c['name']} ({c['id']})")
                for c in cams
            ]),
            label="Camera", required=True,
        )

        # Show current values as defaults if a camera is selected
        selected = ctx.params.get("camera_id")
        cur = {}
        if selected:
            for c in cams:
                if c["id"] == selected:
                    cur = c
                    break

        inputs.str("name", default=cur.get("name", ""),
                   label="Display Name")
        inputs.str("region", default=cur.get("region", ""),
                   label="Region",
                   description="State code or region (e.g. TX, CA)")
        inputs.str("locale_name", default=cur.get("locale", ""),
                   label="Locale",
                   description="Site name (e.g. Store #1042)")
        inputs.str("room", default=cur.get("room", ""),
                   label="Room",
                   description="e.g. lobby, parking-lot, entrance")
        inputs.str("position", default=cur.get("position", ""),
                   label="Position",
                   description="Mount point + orientation")
        inputs.str("notes", default=cur.get("notes", ""),
                   label="Notes",
                   description="Plain English instructions")

        return types.Property(inputs, view=types.View(
            label="Edit Camera"))

    def execute(self, ctx):
        cam_id = ctx.params.get("camera_id")
        if not cam_id:
            return {"updated": False}
        with _mgr._lock:
            cam = _mgr._cams.get(cam_id)
            if not cam:
                return {"updated": False}
            name = ctx.params.get("name", "").strip()
            if name:
                cam["name"] = name
            cam["region"] = ctx.params.get("region", "")
            cam["locale"] = ctx.params.get("locale_name", "")
            cam["room"] = ctx.params.get("room", "")
            cam["position"] = ctx.params.get("position", "")
            cam["notes"] = ctx.params.get("notes", "")
        _save_config()
        return {"updated": True}


class ToggleGridView(foo.Operator):
    """Toggle between grid view and individual camera cards."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="toggle_grid_view", label="Toggle Grid View",
        )

    def execute(self, ctx):
        current = ctx.panel.state.get("_grid_mode", False)
        ctx.panel.state.set({"_grid_mode": not current})
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
        cameras = _mgr.list_cameras()
        state = {
            "_cameras": cameras,
            "_camera_ids": [c["id"] for c in cameras],
        }
        for cam in cameras:
            b64 = _mgr.frame_b64(cam["id"])
            if b64:
                state[f"frame_{cam['id']}"] = (
                    f"data:image/jpeg;base64,{b64}")
        ctx.panel.state.set(state)

    def render(self, ctx):
        panel = types.Object()
        cameras = ctx.panel.state.get("_cameras")
        if cameras is None:
            cameras = _mgr.list_cameras()

        n_cams = len(cameras)
        n_streaming = sum(
            1 for c in cameras if c.get("status") == "streaming")
        n_motion = sum(
            1 for c in cameras if c.get("motion_detected"))
        n_recording = sum(
            1 for c in cameras if c.get("recording_active"))

        # ── status bar ──
        status_parts = [f"**{n_cams}** camera(s)"]
        if n_streaming:
            status_parts.append(f"**{n_streaming}** live")
        if n_motion:
            status_parts.append(f"**{n_motion}** motion")
        if n_recording:
            status_parts.append(f"**{n_recording}** recording")
        panel.str("header", default=" | ".join(status_parts),
                  view=types.MarkdownView())

        # ── primary actions — always visible ──
        panel.btn("add_btn", label="Add Camera",
                  on_click="@securvision/streams/add_camera",
                  variant="contained")
        panel.btn("record_btn", label="Record",
                  on_click="@securvision/streams/toggle_recording")
        panel.btn("snap_btn", label="Snapshot All",
                  on_click="@securvision/streams/snapshot_all_cameras")
        panel.btn("search_btn", label="Search",
                  on_click="@securvision/streams/search_footage",
                  variant="contained")
        panel.btn("timeline_btn", label="Timeline",
                  on_click="@securvision/streams/browse_timeline")
        grid_mode = ctx.panel.state.get("_grid_mode", False)
        panel.btn("grid_btn",
                  label="Card View" if grid_mode else "Grid View",
                  on_click="@securvision/streams/toggle_grid_view")

        # ── timeline section — promoted to top when loaded ──
        timeline_md = ctx.panel.state.get("_timeline_md")
        if timeline_md:
            panel.str("timeline_content", default=timeline_md,
                      view=types.MarkdownView())

        if not cameras:
            panel.view("empty", types.Notice(
                label="No cameras connected. Click Add Camera to "
                      "get started."))
            return types.Property(panel)

        # ── camera feeds ──
        if grid_mode:
            grid_frame = ctx.panel.state.get("_grid_frame")
            if grid_frame:
                panel.img("grid_view", height="600px", width="100%")
            else:
                panel.view("grid_wait", types.Notice(
                    label="Waiting for camera frames..."))
            # Skip individual cards in grid mode — jump to settings
            panel.str("settings_divider",
                      default="---\n**Settings**",
                      view=types.MarkdownView())
            self._render_settings(panel)
            panel.view("refresh_timer", types.TimerView(
                interval=2000,
                on_interval="@securvision/streams/refresh_streams",
            ))
            return types.Property(panel)

        for cam in cameras:
            cam_id = cam["id"]
            status = cam.get("status", "unknown")

            # Build compact status badges
            badges = []
            if status == "streaming":
                badges.append("LIVE")
            elif status == "error":
                badges.append("ERROR")
            elif status in ("connecting", "reconnecting"):
                badges.append(status)

            if cam.get("motion_detected"):
                score = cam.get("motion_score", 0.0)
                azones = cam.get("active_zones", [])
                if azones:
                    badges.append(
                        f"MOTION {score:.0%} [{', '.join(azones)}]")
                else:
                    badges.append(f"MOTION {score:.0%}")

            if cam.get("recording_active"):
                badges.append("REC")
            elif cam.get("recording_mode", "off") != "off":
                badges.append(f"rec:{cam['recording_mode']}")

            det_counts = cam.get("detection_counts", {})
            if det_counts:
                count_str = " ".join(
                    f"{label}:{cnt}"
                    for label, cnt in sorted(det_counts.items()))
                badges.append(count_str)

            badge_str = ("  " + "  ".join(
                f"`{b}`" for b in badges)) if badges else ""

            # Location breadcrumb
            loc_parts = [p for p in (cam.get("region", ""),
                                     cam.get("locale", ""),
                                     cam.get("room", "")) if p]
            loc_line = (" > ".join(loc_parts)) if loc_parts else ""

            card_md = f"**{cam['name']}**{badge_str}"
            if loc_line:
                card_md += f"  \n*{loc_line}*"
            cam_notes = cam.get("notes", "")
            if cam_notes:
                card_md += f"  \n*{cam_notes}*"

            panel.str(f"label_{cam_id}", default=card_md,
                      view=types.MarkdownView())

            src_type = cam.get("source_type", "rtsp")

            if src_type == "hls":
                panel.view(f"player_{cam_id}",
                           types.MediaPlayerView(url=cam["url"]))
            else:
                has_frame = (
                    ctx.panel.state.get(f"frame_{cam_id}") is not None)
                if has_frame:
                    panel.img(f"frame_{cam_id}",
                              height="300px", width="100%")
                else:
                    panel.view(f"wait_{cam_id}", types.Notice(
                        label=f"Connecting to {cam['name']}..."))

        # ── settings — collapsed into a section at the bottom ──
        panel.str("settings_divider",
                  default="---\n**Settings**",
                  view=types.MarkdownView())
        self._render_settings(panel)

        # ── auto-refresh timer (hidden) ──
        panel.view("refresh_timer", types.TimerView(
            interval=2000,
            on_interval="@securvision/streams/refresh_streams",
        ))

        return types.Property(panel)

    @staticmethod
    def _render_settings(panel):
        """Render settings buttons (shared by card and grid modes)."""
        panel.btn("snap_one_btn", label="Snapshot One",
                  on_click="@securvision/streams/snapshot_camera")
        panel.btn("edit_btn", label="Edit Camera",
                  on_click="@securvision/streams/manage_cameras")
        panel.btn("remove_btn", label="Remove Camera",
                  on_click="@securvision/streams/remove_camera",
                  variant="outlined")
        if _check_onvif_deps():
            panel.btn("discover_btn", label="Discover ONVIF",
                      on_click="@securvision/streams/discover_cameras")
        panel.btn("zones_btn", label="Zones",
                  on_click="@securvision/streams/configure_zones")
        panel.btn("motion_cfg_btn", label="Motion",
                  on_click="@securvision/streams/configure_motion")
        panel.btn("record_cfg_btn", label="Recording",
                  on_click="@securvision/streams/configure_recording")
        panel.btn("alerts_btn", label="Alerts",
                  on_click="@securvision/streams/configure_alerts")
        panel.btn("detect_cfg_btn", label="Detection",
                  on_click="@securvision/streams/configure_detection")
        panel.btn("analyze_btn", label="Analyze",
                  on_click="@securvision/streams/analyze_samples")


# ─── Registration ──────────────────────────────────────────────────────


def register(p):
    p.register(AddCamera)
    p.register(RemoveCamera)
    p.register(SnapshotCamera)
    p.register(SnapshotAll)
    p.register(RefreshStreams)
    p.register(DiscoverCameras)
    p.register(ConfigureMotion)
    p.register(ToggleRecording)
    p.register(ConfigureRecording)
    p.register(ConfigureAlerts)
    p.register(ConfigureDetection)
    p.register(AnalyzeSamples)
    p.register(SearchFootage)
    p.register(BrowseTimeline)
    p.register(ManageCameras)
    p.register(ConfigureZones)
    p.register(ImportZonesFromSample)
    p.register(ToggleGridView)
    p.register(CameraStreamsPanel)
