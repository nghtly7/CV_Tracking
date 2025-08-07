import cv2
import json
import os
from deep_sort_realtime.deepsort_tracker import DeepSort

# === CONFIG ===
video_path = "raw_video/out2g1.mp4"
video_name = os.path.splitext(os.path.basename(video_path))[0]
detections_json = f"detection/{video_name}_detections.json"
output_video_path = f"tracked/{video_name}_tracked.mp4"
output_tracks_path = f"tracked/{video_name}_tracks.json"

os.makedirs("tracked", exist_ok=True)

# === INIZIALIZZAZIONE ===
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

# === TRACKER ===
tracker = DeepSort(max_age=30)

# === CARICA DETECTION ===
with open(detections_json, "r") as f:
    all_detections = json.load(f)

# === TRACKING ===
all_tracks = []

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret or frame_id >= len(all_detections):
        break

    detections = all_detections[frame_id]["detections"]

    # Adatta le detection al formato richiesto da DeepSort: [[x, y, w, h], confidence, class_name]
    adapted_dets = []
    for det in detections:
        x1, y1, w, h, conf, class_id = det
        # DeepSort si aspetta: ([x, y, w, h], confidence, class_name)
        adapted_dets.append(([x1, y1, w, h], conf, str(class_id)))

    tracks = tracker.update_tracks(adapted_dets, frame=frame)

    frame_tracks = []
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Disegna il box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        frame_tracks.append({
            "id": int(track_id),
            "bbox": [x1, y1, x2, y2]
        })

    # Scrive nel video e salva i dati
    out.write(frame)
    all_tracks.append({
        "frame": frame_id,
        "tracks": frame_tracks
    })

    frame_id += 1

cap.release()
out.release()

# === SALVA I TRACKS IN JSON ===
with open(output_tracks_path, "w") as f:
    json.dump(all_tracks, f, indent=2)

print(f"[✓] Video tracciato salvato in: {output_video_path}")
print(f"[✓] Tracce salvate in: {output_tracks_path}")
