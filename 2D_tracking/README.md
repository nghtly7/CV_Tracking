# 2D Tracking – Detection, Tracking, Metrics

2D pipeline for football videos based on YOLO for detection and Deep SORT for tracking. Includes evaluation metrics, utilities to extract frames, and checks to verify temporal alignment.

Contents:
- Requirements and installation
- Folder structure
- Input/output formats
- Script configuration
- Execution (Quickstart)
- Utilities
- Tips and troubleshooting

---

## Requirements and installation

Prerequisites:
- Python 3.10+
- Optional GPU (CUDA) for faster inference

Packages:
- ultralytics
- opencv-python
- deep_sort_realtime
- numpy
- pandas
- matplotlib
- motmetrics

Quick install (PowerShell):
- python -m venv .venv
- .\.venv\Scripts\Activate.ps1
- pip install ultralytics opencv-python deep_sort_realtime numpy pandas matplotlib motmetrics

---

## Folder structure

- detection.py
- tracking.py
- metrics.py
- checkFramesCoordination.py
- getVideoFrames.py
- best.pt, best_small.pt           (YOLO weights)
- detection/                        (detection outputs)
  - outX_detections.json
  - outX_annotated.mp4
- tracked/                          (tracking outputs)
  - outX_tracks.json
  - outX_tracked.mp4
- rDetection/, rTracked/            (outputs of rectified data for 3D tracking; same formats as detection/ and tracked/)

Notes:
- Input videos are stored outside this folder (e.g., ../raw_video/outX.mp4).

---

## Input/output formats

Input video:
- Video file (e.g., raw_video/out4.mp4)

Detection (detection/outX_detections.json):
- Per frame:
  - "frame": int
  - "detections": list of [x, y, w, h, conf, class_id] in pixels (XYWH)
- Annotated video: detection/outX_annotated.mp4

Tracking (tracked/outX_tracks.json):
- Per frame:
  - "frame": int
  - "tracks": list of objects:
    - "id": int
    - "bbox": [x1, y1, x2, y2] in pixels (XYXY)
    - "confidence": float
    - "class_id": int
- Video with IDs: tracked/outX_tracked.mp4

Metrics (metrics.py):
- Console report and/or CSV/JSON dumps (see script)
- Typical metrics: MOTA, MOTP (mean IoU), Precision, Recall, F1, ID Switches, FP, FN, Matches

Examples (schematic):

Detection JSON (per frame):
```
{
  "frame": 123,
  "detections": [
    [x, y, w, h, conf, class_id],
    ...
  ]
}
```

Tracking JSON (per frame):
```
{
  "frame": 123,
  "tracks": [
    { "id": 7, "bbox": [x1, y1, x2, y2], "confidence": 0.89, "class_id": 1 },
    ...
  ]
}
```

---

## Script configuration

Detection (detection.py):
- video_path: source video path (e.g., "raw_video/out4.mp4")
- model_path: YOLO weights (e.g., "best.pt" or "best_small.pt")
- conf/thres, device, img_size (if exposed by the script)
- output dir: detection/ (default)

Tracking (tracking.py):
- video_path: same video used in detection
- automatically loads matching detections from detection/
- Deep SORT params (n_init, max_age, max_cosine_distance, nn_budget, etc.) if exposed
- output dir: tracked/

Metrics (metrics.py):
- video: base video name (e.g., "out4")
- tracker/detection file paths if configurable
- evaluation options (frame range, IoU thresholds, classes)

Utilities:
- checkFramesCoordination.py: parameters for sources (video/detections/tracks) to verify
- getVideoFrames.py: input video, output folder, stride/frame range

Open each script and set the GLOBAL VARIABLES at the top of the file.

---

## Execution (Quickstart)

PowerShell (Windows):
- cd c:\Users\nicol\Desktop\CV_Tracking\2D_tracking

1) Detection
- Edit detection.py:
  - video_path = "../raw_video/out4.mp4"
  - model_path = "best.pt"
- Run:
  - python detection.py
- Output:
  - detection/out4_detections.json
  - detection/out4_annotated.mp4

2) Tracking
- Edit tracking.py:
  - video_path = "../raw_video/out4.mp4"
- Run:
  - python tracking.py
- Output:
  - tracked/out4_tracks.json
  - tracked/out4_tracked.mp4

3) Metrics
- Edit metrics.py:
  - video = "out4"
- Run:
  - python metrics.py

---

## Utilities

Extract frames from a video:
- python getVideoFrames.py
- Set input video, frames folder, and interval/stride in the script

Verify frame coordination (GT/detections/tracks):
- python checkFramesCoordination.py
- Useful to ensure frame indices and timestamps are consistent

---

## Tips and troubleshooting

- YOLO models:
  - best_small.pt for modest hardware or faster inference
  - best.pt for higher accuracy
- Classes:
  - Check class_id mapping in the script (e.g., 0=ball, 1=player, 2=referee) and drawing colors
- Bbox formats:
  - Detection: XYWH
  - Tracking: XYXY
  - Avoid mixing them when computing IoU or drawing
- Empty frames at tracking start:
  - Often due to tracker warm-up; use n_init=1 to reduce (if not already set)
- Synchronization:
  - If common frames GT↔tracker are 0, check mapping or frame offsets
- Performance:
  - Use --device 0 (if supported) or set device="cuda" in the script for GPU
- Library versions:
  - If ultralytics changes API, verify YOLO(model).predict calls
- rDetection/rTracked outputs:
  - Folders for rectified/alternative runs; same file formats as detection/ and tracked/

---

## Quick commands

- python detection.py
- python tracking.py
- python metrics.py
- python getVideoFrames.py