# CV_Tracking – Project overview, structure and quickstart

Repository for 2D and 3D tracking on football videos. It includes:
- 2D detection and tracking (YOLO + Deep SORT)
- 3D triangulation and tracking from multi-view 2D inputs
- Visualization, metrics, and Unreal Engine export

Module details are in:
- 2D: 2D_tracking/README.md
- 3D: 3D_tracking/README.md

---

## Project structure

```
CV_Tracking/
├─ 2D_tracking/
│  ├─ detection.py                 # YOLO detections
│  ├─ tracking.py                  # Deep SORT tracking
│  ├─ metrics.py                   # 2D evaluation utilities
│  ├─ getVideoFrames.py            # extract frames
│  ├─ checkFramesCoordination.py   # frame/time alignment checks
│  ├─ best.pt, best_small.pt       # YOLO weights
│  ├─ detection/                   # detection outputs (JSON/annotated MP4)
│  ├─ tracked/                     # tracking outputs (JSON/MP4)
│  ├─ rDetection/, rTracked/       # rectified outputs for 3D (same formats)
│  └─ README.md
│
├─ 3D_tracking/
│  ├─ triangulation.py             # multi-view association + triangulation
│  ├─ 3D_tracker.py                # 3D tracking (players/ball)
│  ├─ 3dMetrics.py                 # 3D metrics
│  ├─ displayData.py               # 3D viewer (CSV in meters, Z up)
│  ├─ export_unreal_engine.py      # export to Unreal (cm, Z-up)
│  ├─ triangulations/              # per-frame associations/triangulated data
│  ├─ tracks3d/                    # 3D tracks CSV + stats
│  ├─ unreal/                      # Unreal CSV/JSONL outputs
│  └─ README.md
│
├─ datasetManipulation/            # dataset utilities (training/analysis)
│  ├─ datasetStats.py              # stats/EDA on labeled datasets
│  ├─ dataSplit.py                 # train/val/test split
│  └─ labelCorrection.py           # label fixing/transforms
│
├─ GroundTruthData/                # labeled data (COCO) and notes
│  ├─ README.dataset.txt
│  ├─ README.roboflow.txt
│  └─ train/
│     ├─ _annotations.coco.json
│     ├─ _annotations_rectified.coco.json
│     └─ *.jpg                     # training images
│
├─ support_material/
│  ├─ GT_camera_positions.png
│  └─ 3D_tracking_material/
│     ├─ rectified_videos.py       # generate rectified videos
│     ├─ rectified_GTjson.py       # rectify GT JSON to field coords
│     └─ camera_data/
│        ├─ cam_2/ cam_4/ cam_13/  # per-camera bundles
│        │  ├─ metadata.json
│        │  ├─ calib/              # camera_calib.json, img_points.json
│        │  └─ dump/               # calibration logs/snapshots
│
├─ raw_video/                      # original input videos
│  ├─ out2.mp4 out4.mp4 out13.mp4  # + optional *_g1.mp4
│
├─ rectified_video/                # rectified videos (for 3D)
│  ├─ out2.mp4 out4.mp4 out13.mp4
│
└─ README.md                       # this file
```

Conventions
- Coordinates: meters (Z up) in 3D; pixels in 2D.
- Classes: player, referee, ball. Scripts accept both names and IDs (1=player, 2=referee, 0=ball).
- Scripts use global CONFIG variables at the top. Adjust paths/parameters before running.

---

## Requirements

Recommended
- Windows, Python 3.10+ (tested on 3.12)
- Optional CUDA GPU for faster 2D inference

Packages
- Core: numpy, pandas, matplotlib, opencv-python
- 2D: ultralytics, deep_sort_realtime, motmetrics
- 3D: scipy

Virtual environment (PowerShell)
- python -m venv .venv
- .\.venv\Scripts\Activate.ps1
- pip install numpy pandas matplotlib opencv-python scipy ultralytics deep_sort_realtime motmetrics

---

## Quickstart

Before you start
- Put your raw videos in raw_video/.
- If needed for 3D, generate rectified videos in rectified_video/ using support_material/3D_tracking_material/rectified_videos.py.
- Open each script you will run and set the CONFIG section (paths, FPS, etc.).
```
cd 2D_tracking
```
2D pipeline (per video)
1) Detections
- Edit 2D_tracking/detection.py (video_path, model_path)
- Run: python 2D_tracking/detection.py
- Outputs: 2D_tracking/detection/

2) Tracking
- Edit 2D_tracking/tracking.py (same video_path)
- Run: python 2D_tracking/tracking.py
- Outputs: 2D_tracking/tracked/

3) (Optional) Metrics and utilities
- Metrics: python 2D_tracking/metrics.py
- Frames: python 2D_tracking/getVideoFrames.py
- Frame/time checks: python 2D_tracking/checkFramesCoordination.py

Details: see 2D_tracking/README.md

3D pipeline (multi-view)
Prerequisites
- Per-camera detections/tracks rectified to field space in 2D_tracking/rDetection and 2D_tracking/rTracked.
- Camera intrinsics/extrinsics in support_material/3D_tracking_material/camera_data.
```
cd 3D_tracking
```
Steps
1) Triangulation
- Edit 3D_tracking/triangulation.py (camera params, inputs)
- Run: python 3D_tracking/triangulation.py
- Outputs: 3D_tracking/triangulations/

2) 3D tracking
- Edit 3D_tracking/3D_tracker.py (model params, FPS)
- Run: python 3D_tracking/3D_tracker.py
- Outputs: 3D_tracking/tracks3d/

3) Visualization
- Edit 3D_tracking/displayData.py (CSV_PATH, FPS, FIELD_SIZE)
- Run: python 3D_tracking/displayData.py

4) Unreal export
- Edit 3D_tracking/export_unreal_engine.py (INPUT_CSV, OUT_DIR, WORLD_ROT_DEG, WORLD_OFFSET_M, …)
- Run: python 3D_tracking/export_unreal_engine.py
- Outputs: 3D_tracking/unreal/

Details: see 3D_tracking/README.md

---

## Outputs overview

2D
- detection/: per-frame detections (JSON) and annotated videos (MP4)
- tracked/: per-frame tracks (JSON) and ID-overlay videos (MP4)
- rDetection/, rTracked/: rectified outputs for 3D

3D
- triangulations/: per-frame multi-view associations/points
- tracks3d/: consolidated 3D tracks CSV + stats
- unreal/: unreal_tracks.csv and unreal_frames.jsonl

Datasets
- GroundTruthData/: COCO annotations and images for training/evaluation
- datasetManipulation/: helpers for splits, corrections, and dataset stats

---

## Tips

- Adjust CONFIG blocks before running any script.
- Keep consistent names (e.g., out2/out4/out13) across raw, rectified, 2D, and 3D outputs.
- For camera setup and rectification, check support_material/3D_tracking_material/.
- For module-specific usage, formats, and troubleshooting, read:
  - 2D_tracking/README.md
  - 3D_tracking/README.md
