# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FiftyOne plugin (`@securvision/streams`) — a live camera streaming NVR with ML-powered analysis for the FiftyOne App. 15 operators, 1 panel. Features: camera management with 4-level location hierarchy (Region/Locale/Room/Position), ONVIF auto-discovery, motion detection, MP4 recording with pre/post-roll, email alerts, object detection (YOLO/model zoo), footage search by date+location+label, timeline browsing, and persistent config that survives restarts. Apache 2.0.

## Development Setup

```bash
pip install -r requirements.txt                      # core deps
pip install onvif-zeep WSDiscovery                   # optional: ONVIF discovery
pip install 'fiftyone[zoo]' torch ultralytics        # optional: object detection
ln -s "$(pwd)" "$(fiftyone config plugins_dir)/fiftyone-securvision-streams"
fiftyone app launch
```

No build step, linter, or test suite. Manual testing through the FiftyOne App UI.

## Architecture

Single `__init__.py` (~2450 lines):

1. **Persistent Config** — `securvision_config.json` in the plugin directory. `_save_config()` writes after every mutation. `_auto_reconnect()` restores cameras + settings on startup. Schema: `{cameras: {id: {name, url, source_type, region, locale, room, position, motion_config, recording_config, recording_mode, alerts_enabled}}, alerts: {...}, detection: {...}}`.

2. **Detection Model Management** — Lazy `_get_detection_model()` with thread-safe caching. FiftyOne model zoo default (`yolov8s-coco-torch`), custom model override via config. `_check_detection_deps()` for graceful degradation.

3. **Recording Index** — Background `_recording_indexer_loop` thread drains a `queue.Queue` of completed segments into `fo.Sample` objects. Decouples capture threads from FiftyOne DB.

4. **StreamManager** — Thread-safe singleton, one daemon thread per camera. Pipeline: read → JPEG encode → motion detect → record → alert → update state. Location fields (region/locale/room/position) flow through all 13 touchpoints: cam dict → list_cameras → snapshot samples → recording indexer → close_writer enqueue → operator forms → search queries → panel cards → timeline.

5. **Capture classes** — `_DemoCapture`, `_VideoFileCapture`, `_HttpSnapshotCapture`, `_BrowserOnlyCapture`

6. **Operators** (15):
   - Camera: `AddCamera`, `RemoveCamera`, `DiscoverCameras`, `ManageCameras` (edit location)
   - Snapshot: `SnapshotCamera`, `SnapshotAll`
   - NVR: `ConfigureMotion`, `ToggleRecording`, `ConfigureRecording`, `ConfigureAlerts`
   - Detection: `ConfigureDetection`, `AnalyzeSamples`
   - Search: `SearchFootage` (region/locale/room + date + label + confidence + motion filters), `BrowseTimeline`
   - Internal: `RefreshStreams`

7. **Panel** — Clean hierarchy: 5 primary buttons (Add, Record, Snapshot, Search, Timeline) → camera cards with badge indicators + location breadcrumb → settings section at bottom → timeline section when loaded.

## Key Design Decisions

- **Location hierarchy** flows through every sample: region (state), locale (site), room (area), position (mount point). Enables queries like "all parking lot cameras in Texas."
- **Persistent config** saves to `securvision_config.json` after every mutation. Auto-reconnects all cameras on plugin load.
- **Panel UX**: 5 primary actions on top, settings collapsed at bottom. Camera cards show `**Name** \`LIVE\` \`MOTION 12%\` \`REC\`  \n*TX > Store #1042 > parking-lot*`
- Detection model lazy-loaded, `fiftyone.zoo` optional
- `SearchFootage` builds chained `DatasetView`, applies via `ctx.trigger("@voxel51/operators/set_view")`
- Recording indexer uses queue to avoid FiftyOne DB calls in capture threads
- Pre-roll buffer only active in "motion" mode
- HLS excluded from snapshot/recording

## Plugin Manifest

`fiftyone.yml`: version 1.0.0, FiftyOne >= 0.25, 15 operators, 1 panel.
