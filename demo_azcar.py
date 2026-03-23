"""Demo: Load all Arizona emissions testing station cameras.

Writes securvision_config.json so the plugin auto-connects 18 public
webcams from https://www.myazcar.com/wait-times on next FiftyOne launch.

Usage:
    python demo_azcar.py
    fiftyone app launch
"""

import json
import os

BASE_URL = "https://www.azvecportal.com/queuecam"

# Phoenix Metro cameras
PHOENIX_CAMERAS = [
    ("M01", "Beverly", "5850 W. Beverly Ln., Glendale",
     "lobby", "Front entrance, facing queue"),
    ("M02", "110th Ave", "7140 N. 110th Ave., Glendale",
     "lobby", "Main queue area"),
    ("M03", "23rd Ave (Waiver)", "10210 N. 23rd Ave., Phoenix",
     "lobby", "Waiver/Referee station"),
    ("M04", "Roosevelt", "5302 W. Roosevelt St., Phoenix",
     "lobby", "Front entrance, facing queue"),
    ("M06", "Madison (Waiver)", "4949 E. Madison St., Phoenix",
     "lobby", "Waiver/Referee station"),
    ("M07", "1st Ave", "1851 W. 1st Ave., Mesa",
     "lobby", "Main queue area"),
    ("M08", "Ivy", "4442 E. Ivy St., Mesa",
     "lobby", "Front entrance, facing queue"),
    ("M09", "Beck", "20 N. Beck Ave., Chandler",
     "lobby", "Main queue area"),
    ("M10", "Evans", "8448 E. Evans, Scottsdale",
     "lobby", "Front entrance, facing queue"),
    ("M12", "Riverview", "1520 E. Riverview Dr., Phoenix",
     "lobby", "Main queue area"),
    ("M13", "Airport", "2360 S. Airport Blvd., Chandler",
     "lobby", "Main queue area"),
    ("M14", "Westgate", "13425 W. Westgate Dr., Surprise",
     "lobby", "Front entrance, facing queue"),
    ("M15", "Deer Valley", "501 W. Deer Valley Rd., Phoenix",
     "lobby", "Main queue area"),
    ("M16", "38th Ave", "565 E. 38th Ave., Apache Junction",
     "lobby", "Main queue area"),
    ("M17", "Eddie Albert", "16140 W. Eddie Albert Way, Goodyear",
     "lobby", "Front entrance, facing queue"),
]

# Tucson cameras
TUCSON_CAMERAS = [
    ("P01", "Stocker", "1301 S. Stocker Dr., Tucson",
     "lobby", "Main queue area"),
    ("P02", "Business Center (Waiver)", "3931 N. Business Center Dr., Tucson",
     "lobby", "Waiver/Referee station"),
    ("P03", "Renaissance", "6661 S. Renaissance Dr., Tucson",
     "lobby", "Front entrance, facing queue"),
]


def build_camera_configs():
    """Build camera config dicts for all AZ emissions stations."""
    configs = []
    for cam_id, name, address, room, position in PHOENIX_CAMERAS:
        configs.append({
            "camera_id": cam_id.lower(),
            "name": f"AZ Emissions - {name}",
            "url": f"{BASE_URL}/{cam_id}.jpg",
            "source_type": "http_snapshot",
            "region": "AZ",
            "locale": f"Emissions - {name} ({address})",
            "room": room,
            "position": position,
        })
    for cam_id, name, address, room, position in TUCSON_CAMERAS:
        configs.append({
            "camera_id": cam_id.lower(),
            "name": f"AZ Emissions - {name}",
            "url": f"{BASE_URL}/{cam_id}.jpg",
            "source_type": "http_snapshot",
            "region": "AZ",
            "locale": f"Emissions - {name} ({address})",
            "room": room,
            "position": position,
        })
    return configs


def main():
    configs = build_camera_configs()
    print(f"SecurVision Demo: {len(configs)} Arizona emissions station cameras\n")

    for cfg in configs:
        print(f"  {cfg['camera_id']:5s}  {cfg['name']:40s}  {cfg['url']}")

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "securvision_config.json")

    cameras = {}
    for cfg in configs:
        cameras[cfg["camera_id"]] = {
            "name": cfg["name"],
            "url": cfg["url"],
            "source_type": cfg["source_type"],
            "region": cfg["region"],
            "locale": cfg["locale"],
            "room": cfg["room"],
            "position": cfg["position"],
            "notes": f"AZ emissions station - {cfg['position']}",
            "custom_zones": [],
            "zones": [[True, True, True] for _ in range(3)],
            "zone_names": [
                ["top-left", "top-center", "top-right"],
                ["mid-left", "center", "mid-right"],
                ["bottom-left", "bottom-center", "bottom-right"],
            ],
            "motion_config": {
                "threshold": 30, "min_area": 800, "enabled": True,
            },
            "recording_config": {
                "output_dir": "securvision_recordings",
                "segment_seconds": 300,
                "preroll_seconds": 5,
                "postroll_seconds": 10,
                "fps": 2,  # HTTP snapshots are slow, 2fps is plenty
            },
            "recording_mode": "off",
            "alerts_enabled": False,
        }

    saved_config = {
        "cameras": cameras,
        "alerts": {
            "enabled": False, "smtp_host": "", "smtp_port": 587,
            "smtp_tls": True, "sender": "", "password": "",
            "recipients": [], "cooldown_seconds": 300,
        },
        "detection": {
            "model_name": "yolov8s-coco-torch",
            "confidence_threshold": 0.25,
            "label_field": "detections",
            "auto_detect_snapshots": False,
            "overlay_enabled": False,
            "overlay_interval": 5,
        },
    }

    with open(config_path, "w") as f:
        json.dump(saved_config, f, indent=2)

    print(f"\nConfig written to: {config_path}")
    print(f"Total cameras: {len(configs)}")
    print("\nNext steps:")
    print("  1. fiftyone app launch")
    print("  2. Open the Camera Streams panel")
    print("  3. All 18 cameras will auto-connect from saved config")
    print("  4. Use Search Footage to filter by region=AZ")
    print("  5. Use Browse Timeline to see activity")
    print("  6. Use Analyze Samples after taking snapshots to run detection")


if __name__ == "__main__":
    main()
