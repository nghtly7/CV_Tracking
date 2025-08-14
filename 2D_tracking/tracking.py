import cv2
import json
import os
from deep_sort_realtime.deepsort_tracker import DeepSort

# === CONFIG ===
video_path = "../raw_video/out13.mp4"
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
tracker = DeepSort(max_age=30, n_init=1)  # conferma dopo 1 hit

# === CARICA DETECTION ===
with open(detections_json, "r") as f:
    all_detections = json.load(f)

# Cache ultimi valori validi per ogni track_id
last_conf = {}   # track_id -> float
last_class = {}  # track_id -> int

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
    det_class_map = {}  # Dizionario per mappare detection a class_id
    
    for det_idx, det in enumerate(detections):
        x1, y1, w, h, conf, class_id = det
        adapted_dets.append(([x1, y1, w, h], conf, str(class_id)))
        # Salva la mappatura tra l'indice della detection e il class_id
        det_class_map[det_idx] = int(class_id)

    tracks = tracker.update_tracks(adapted_dets, frame=frame)

    frame_tracks = []
    for track in tracks:
        # salva solo tracce aggiornate nel frame e confermate
        if track.time_since_update != 0:
            continue
        if not track.is_confirmed():
            continue

        # valori di detection devono essere presenti se aggiornato
        det_conf = track.get_det_conf()
        det_cls = track.get_det_class()
        if det_conf is None or det_cls is None:
            continue

        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # clamp ai bordi del frame
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))

        # class id e conf dal frame corrente
        try:
            class_id = int(det_cls)
        except (ValueError, TypeError):
            continue  # evita valori non validi

        conf = float(det_conf)

        # Colori fissi per classe: 0=verde, 1=azzurro(cyan), 2=rosso
        color_map = {
            0: (0, 255, 0),      # BGR: verde
            1: (255, 255, 0),    # BGR: azzurro/cyan
            2: (0, 0, 255),      # BGR: rosso
        }
        color = color_map.get(class_id, (200, 200, 200))  # fallback grigio
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID {int(track.track_id)} | C {class_id} | {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = y1 - 8 if y1 - 8 > 10 else (y1 + th + 8)

        # background per il testo (rettangolo pieno)
        pt1 = (int(x1), int(y_text - th - 4))
        pt2 = (int(x1 + tw + 4), int(y_text + 2))
        cv2.rectangle(frame, pt1, pt2, color, -1)

        # testo
        cv2.putText(frame, label, (int(x1 + 2), int(y_text)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        frame_tracks.append({
            "id": int(track.track_id),
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "class_id": class_id,
        })

    # Scrive nel video e salva i dati
    out.write(frame)
    all_tracks.append({"frame": frame_id, "tracks": frame_tracks})

    frame_id += 1

cap.release()
out.release()

# === SALVA I TRACKS IN JSON ===
with open(output_tracks_path, "w") as f:
    json.dump(all_tracks, f, indent=2)

print(f"[✓] Video tracciato salvato in: {output_video_path}")
print(f"[✓] Tracce salvate in: {output_tracks_path}")
