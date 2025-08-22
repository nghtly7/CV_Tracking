# 3D Tracking – Triangulation, Tracking, Viewer, Metrics, Unreal Export

Complete 3D tracking pipeline for sports: from multi-camera 2D observations to field-plane metrics, visualization, and export for Unreal Engine.

Contents:
- Folder structure and prerequisites
- Data and conventions
- End-to-end pipeline
- Script configuration
- Execution (Quickstart)
- Input/output formats
- Tips and troubleshooting

---

## Folder structure and prerequisites

Typical contents of 3D_tracking:
- triangulation.py
- 3D_tracker.py
- 3dMetrics.py
- displayData.py
- export_unreal_engine.py
- tracks3d/
  - tracks3d.csv
  - stats.json
- triangulations/
  - associations_#.json, triangulated_#.json (per frame)
- unreal/
  - unreal_tracks.csv, unreal_frames.jsonl (Unreal export outputs)

Requirements (Python 3.10+ recommended):
- numpy, pandas, matplotlib, scipy, opencv-python (for triangulation/utilities)

Quick install (PowerShell):
- pip install numpy pandas matplotlib scipy opencv-python

---

## Data and conventions

- 3D space:
  - Units: meters
  - Z up; XY is the field plane
- Classes:
  - Canonical: player, referee, ball
  - Mapping also supported from IDs: 1→player, 2→referee, 0→ball
- 3D predictions CSV (tracks3d.csv):
  - Typical columns: t, track_id, class, x, y, z, vx, vy, vz, meas_err_px
  - Column aliases (t|frame, x|x_m, …) are accepted where supported by scripts

---

## End-to-end pipeline

1) Triangulation (triangulation.py)
- Load calibrations (K, R, t) and rectified 2D observations
- Epipolar matching and multi-view clustering
- Triangulation (DLT) + non-linear refine (LM) + covariance estimate
- Filters: reprojection error, min number of views
- Per-frame output in triangulations/

2) 3D Tracking (3D_tracker.py)
- Filter and deduplicate triangulations
- CV model for player/referee; ballistic model for ball
- Association via Mahalanobis distance and chi^2 gating
- Track confirm/miss/termination logic
- Outputs: tracks3d/tracks3d.csv and stats.json

3) 3D Viewer (displayData.py)
- Frame-by-frame playback with keyboard controls
- Field drawing (data bounding box or preset dimensions)
- Optional direction arrows (yaw) for player/referee

4) 3D Metrics (3dMetrics.py)
- Project GT to field plane via homographies
- Temporal alignment GT→pred (FRAME_SCALE/OFFSET)
- Optional GT dedup and pred post-processing
- Metric matching and computation (detection/position, optional tracking)

5) Export to Unreal Engine (export_unreal_engine.py)
- Meters→centimeters conversion, world rotation/offset, yaw estimation
- CSV for DataTable/Blueprint/Sequencer + per-frame JSONL

---

## Script configuration

Triangulation (triangulation.py)
- Paths to calibrations and observations
- Thresholds for reprojection error, min views

3D Tracking (3D_tracker.py)
- Model parameters (noises, gravity), chi^2 gating
- Dedup thresholds and confirm/miss handling
- FPS

3D Viewer (displayData.py)
- CSV_PATH: path to CSV (meters, Z up)
- FPS: fps for time bar
- FIELD_SIZE: None or (LENGTH, WIDTH) in meters

3D Metrics (3dMetrics.py)
- TRACKS3D_CSV, COCO_GT_PATH, CAMERA_DATA
- FRAME_SCALE, FRAME_OFFSET
- Gating and thresholds for matching and GT dedup
- Enable post-processing (stitch/speed/smooth/NMS) and tracking metrics

Unreal Export (export_unreal_engine.py)
- INPUT_CSV: input CSV (meters)
- OUT_DIR: output folder (e.g., 3D_tracking/unreal)
- FPS_TRACKS: track fps (for time_s)
- CM_PER_M: 100.0 (UE uses cm)
- WORLD_ROT_DEG: counterclockwise XY-plane rotation (degrees)
- WORLD_OFFSET_M: (x, y, z) in meters
- REBASE_TIME: if True, time_s starts from min(t)
- WRITE_JSONL, SORT_OUTPUT

---

## Execution (Quickstart)

PowerShell (Windows):
- cd c:\Users\nicol\Desktop\CV_Tracking\3D_tracking

Triangulation:
- python triangulation.py

3D Tracking:
- python 3D_tracker.py

Viewer:
- python displayData.py
- Keys: SPACE (play/pause), ←/→ (frame), ↑/↓ (speed), Q (quit)

Metrics:
- python 3dMetrics.py

Unreal Export:
- python export_unreal_engine.py
- Output: 3D_tracking/unreal

---

## Input/output formats

3D predictions (tracks3d/tracks3d.csv)
- t (int), track_id (int), class (str|int), x, y, z (float, meters)
- Optional: vx, vy, vz, meas_err_px

Unreal Export
- unreal_tracks.csv (centimeters, Z up)
  - row_name: "{class}_{track_id}_{frame}"
  - track_id, class, frame, time_s, x_cm, y_cm, z_cm, yaw_deg
- unreal_frames.jsonl (optional, 1 line per frame)
  - For frame t: { "frame": t, "time_s": ..., "objects": [ { "id", "class", "x", "y", "z", "yaw" }, ... ] }

3D Viewer
- Reads tracks3d.csv (column aliases supported)
- Plots 3 scatter per class + quiver for direction

Metrics (3dMetrics.py)
- metrics_summary.json with detection metric (by thresholds), position (MAE/RMSE/percentiles), optional tracking (CLEAR-MOT/IDF1)

---

## Tips and troubleshooting

- Columns/aliases:
  - Scripts accept aliases for t|frame, track_id|id, class|label|category, x|x_m, y|y_m, z|z_m
- Classes:
  - Accepts both names and IDs; internal mapping normalizes to player/referee/ball
- Yaw:
  - Computed from direction in XY plane in world space; if object is stationary → yaw=0
- Unreal:
  - Set WORLD_ROT_DEG/WORLD_OFFSET_M to align coordinates at UE map level
  - UE uses centimeters; output is already converted
- Viewer:
  - If FIELD_SIZE is not specified, uses XY bounding box of data
- Metrics alignment:
  - FRAME_SCALE/OFFSET must reflect GT↔pred FPS ratio (e.g. GT 5 fps vs pred 25fps)
- Troubleshooting:
  - Common errors include:
    - Incorrect FRAME_SCALE/OFFSET → wrong temporal alignment
    - Inconsistent calibrations → triangulation/association errors
    - Too tight gating parameters → missing or fragmented tracks
    - Unrecognized column aliases → CSV reading error
  - Check output logs for error or warning messages
  - Visually verify results with 3D viewer
  - Step-by-step debug to isolate issues in specific scripts or functions

---

## Design notes

- Camera ID normalization: outN → cam_N (regex/normalizer).
- GT reference points:
  - Player/Referee: bottom-center bbox (ground point).
  - Ball: center bbox (can be in the air).
- Stability/performance:
  - H^(-1) cache for camera, conservative thresholds for stitching and NMS.
- Some parts may require code completion/refinement where present