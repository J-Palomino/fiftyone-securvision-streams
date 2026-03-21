# SecurVision Streams

A [FiftyOne](https://github.com/voxel51/fiftyone) plugin for streaming live cameras directly in the FiftyOne App.

Connect RTSP cameras, HLS streams, HTTP snapshot endpoints, video files, or local webcams — view them in a live panel and save snapshots to your dataset for labeling.

## Install

```bash
fiftyone plugins download https://github.com/J-Palomino/fiftyone-securvision-streams
```

Or clone and symlink for development:

```bash
git clone https://github.com/J-Palomino/fiftyone-securvision-streams.git
ln -s "$(pwd)/fiftyone-securvision-streams" \
      "$(fiftyone config plugins_dir)"/securvision-streams
```

Install Python dependencies:

```bash
pip install -r fiftyone-securvision-streams/requirements.txt
```

## Usage

1. Launch the FiftyOne App with any dataset loaded
2. Open the **Camera Streams** panel (click `+` in the panel tabs)
3. Click **Add Camera** and fill in the form
4. Frames auto-refresh every 2 seconds
5. Click **Snapshot All** to save current frames to the dataset

## Supported sources

| Source type | Description | How it works |
|---|---|---|
| **RTSP** | `rtsp://host:554/path` | OpenCV + FFmpeg, TCP forced, 10s open timeout, 5s read timeout, buffer size 1 |
| **HLS / Web** | `https://host/stream.m3u8` | Plays directly in the browser via `MediaPlayerView` — no server-side decode |
| **HTTP Snapshot** | `http://camera/snap.jpg` | Polls the URL on each frame, decodes the JPEG response |
| **Video File** | `/path/to/video.mp4` | Loops playback via OpenCV |
| **Webcam** | Device index (`0`, `1`, …) | Local USB / built-in camera |
| **Demo** | Any string | Synthetic security camera frames for testing without hardware |

## Operators

| Operator | Description |
|---|---|
| `add_camera` | Form UI to connect a new camera source |
| `remove_camera` | Dropdown to disconnect a camera |
| `snapshot_camera` | Save one camera's frame to the dataset |
| `snapshot_all_cameras` | Save all streaming cameras' frames |
| `refresh_streams` | (internal) Called by the panel timer to push frames to state |

## Panel

The **Camera Streams** panel shows:

- Camera status (connecting, streaming, reconnecting, error)
- Live frames as base64 images (RTSP, webcam, video, HTTP snapshot, demo)
- Native browser video player for HLS/DASH streams
- Action buttons for add, remove, snapshot

Frames auto-refresh every 2 seconds via `TimerView`. RTSP streams auto-reconnect with exponential backoff (up to 5 attempts) on connection loss.

## Requirements

- FiftyOne >= 0.25
- Python 3.9+
- OpenCV (`opencv-python >= 4.5`)
- NumPy
- Requests

## License

Apache 2.0
