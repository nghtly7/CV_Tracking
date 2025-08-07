from ultralytics import YOLO
import os

# Carica il tuo modello YOLOv8 fine-tuned (il file best.pt che hai ottenuto dall'addestramento)
model = YOLO('best_small.pt')

# Percorsi dei video o delle cartelle di frame
video_paths = {
    'out2': 'raw_video/out2.mp4',
    'out4': 'raw_video/out4.mp4',
    'out13': 'raw_video/out13.mp4',
    # Esempio per cartella di frame:
    # 'out2_frames': 'frames/out2/'
}

# Dizionario per memorizzare i risultati del tracking per ogni angolazione
all_tracking_results = {}

for angle_name, source_path in video_paths.items():
    print(f"Starting 2D tracking for {angle_name}...")

    # Se il percorso è una cartella, controlla che esistano immagini
    if os.path.isdir(source_path):
        images = [f for f in os.listdir(source_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            print(f"Nessuna immagine trovata nella cartella {source_path}. Salto {angle_name}.")
            continue
        tracking_source = source_path
    else:
        tracking_source = source_path

    results = model.track(source=tracking_source, conf=0.3, iou=0.5, persist=True, tracker='bytetrack.yaml',show=True, save=True)

    angle_tracking_data = []
    for i, r in enumerate(results):
        if r.boxes.id is not None:
            frame_data = {
                'frame_id': i,
                'objects': []
            }
            for box, track_id, conf, cls_id in zip(r.boxes.xyxy, r.boxes.id, r.boxes.conf, r.boxes.cls):
                frame_data['objects'].append({
                    'id': int(track_id),
                    'class': model.names[int(cls_id)],
                    'bbox_xyxy': box.tolist(),
                    'confidence': float(conf)
                })
            angle_tracking_data.append(frame_data)
        else:
            angle_tracking_data.append({'frame_id': i, 'objects': []}) # Nessun oggetto tracciato in questo frame

    all_tracking_results[angle_name] = angle_tracking_data
    print(f"Finished 2D tracking for {angle_name}.")

# Ora 'all_tracking_results' contiene tutte le traiettorie 2D per ciascuna angolazione
# Puoi salvarle in un formato JSON o CSV per un uso futuro.
import json
with open('2d_tracking_results.json', 'w') as f:
    json.dump(all_tracking_results, f, indent=4)