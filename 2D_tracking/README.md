# 2D Tracking – Detection, Tracking, Metriche

Pipeline 2D per video di calcio basata su YOLO per la detection e Deep SORT per il tracking. Include metriche di valutazione, utility per estrarre frame e per verificare il corretto allineamento temporale.

Contenuti:
- Requisiti e installazione
- Struttura cartelle
- Formati di input/output
- Configurazione degli script
- Esecuzione (Quickstart)
- Utility
- Suggerimenti e troubleshooting

---

## Requisiti e installazione

Prerequisiti:
- Python 3.10+
- GPU opzionale (CUDA) per inferenza più veloce

Pacchetti:
- ultralytics
- opencv-python
- deep_sort_realtime
- numpy
- pandas
- matplotlib
- motmetrics

Installazione rapida (PowerShell):
- python -m venv .venv
- .\.venv\Scripts\Activate.ps1
- pip install ultralytics opencv-python deep_sort_realtime numpy pandas matplotlib motmetrics

---

## Struttura cartelle

- detection.py
- tracking.py
- metrics.py
- checkFramesCoordination.py
- getVideoFrames.py
- best.pt, best_small.pt           (pesi YOLO)
- detection/                        (output detection)
  - outX_detections.json
  - outX_annotated.mp4
- tracked/                          (output tracking)
  - outX_tracks.json
  - outX_tracked.mp4
- rDetection/, rTracked/            (cartelle di output per dati rettificati, necessari per il tracking 3d, stessi formati di detection/ e tracked/)

Note:
- I video di input sono esterni a questa cartella (es. ../raw_video/outX.mp4).

---

## Formati di input/output

Input video:
- File video (es. raw_video/out4.mp4)

Detection (detection/outX_detections.json):
- Per frame:
  - "frame": int
  - "detections": lista di [x, y, w, h, conf, class_id] in pixel (XYWH)
- Annotated video: detection/outX_annotated.mp4

Tracking (tracked/outX_tracks.json):
- Per frame:
  - "frame": int
  - "tracks": lista di oggetti:
    - "id": int
    - "bbox": [x1, y1, x2, y2] in pixel (XYXY)
    - "confidence": float
    - "class_id": int
- Video con ID: tracked/outX_tracked.mp4

Metriche (metrics.py):
- Report a console e/o salvataggi CSV/JSON (vedi script)
- Metri che tipiche: MOTA, MOTP (IoU medio), Precision, Recall, F1, ID Switches, FP, FN, Matches

Esempi (schematici):

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

## Configurazione degli script

Detection (detection.py):
- video_path: percorso al video sorgente (es. "raw_video/out4.mp4")
- model_path: pesi YOLO (es. "best.pt" o "best_small.pt")
- conf/thres, device, img_size (se previsti nello script)
- output dir: detection/ (predefinita)

Tracking (tracking.py):
- video_path: stesso video usato in detection
- carica automaticamente detection corrispondenti da detection/
- parametri Deep SORT (n_init, max_age, max_cosine_distance, nn_budget, ecc.) se esposti
- output dir: tracked/

Metriche (metrics.py):
- video: nome base del video (es. "out4")
- percorsi file tracker/detection se configurabili
- opzioni di valutazione (range frame, soglie IoU, classi)

Utility:
- checkFramesCoordination.py: parametri per sorgenti (video/detections/tracks) da verificare
- getVideoFrames.py: input video, cartella output, stride/frame range

Apri ogni script e imposta le VARIABILI GLOBALI all’inizio del file.

---

## Esecuzione (Quickstart)

PowerShell (Windows):
- cd c:\Users\nicol\Desktop\CV_Tracking\2D_tracking

1) Detection
- Modifica in detection.py:
  - video_path = "../raw_video/out4.mp4"
  - model_path = "best.pt"
- Esegui:
  - python detection.py
- Output:
  - detection/out4_detections.json
  - detection/out4_annotated.mp4

2) Tracking
- Modifica in tracking.py:
  - video_path = "../raw_video/out4.mp4"
- Esegui:
  - python tracking.py
- Output:
  - tracked/out4_tracks.json
  - tracked/out4_tracked.mp4

3) Metriche
- Modifica in metrics.py:
  - video = "out4"
- Esegui:
  - python metrics.py

---

## Utility

Estrazione frame dal video:
- python getVideoFrames.py
- Imposta input video, cartella frames e intervallo/stride nello script

Verifica coordinazione frame (GT/detections/tracks):
- python checkFramesCoordination.py
- Utile per assicurarci che gli indici di frame e i tempi siano coerenti

---

## Suggerimenti e troubleshooting

- Modelli YOLO:
  - best_small.pt per hardware più modesto o inferenza rapida
  - best.pt per accuratezza maggiore
- Classi:
  - Verifica la mappatura class_id nello script (es. 0=ball, 1=player, 2=referee) e i colori impostati
- Formati bbox:
  - Detection: XYWH
  - Tracking: XYXY
  - Evita confusioni quando calcoli IoU o quando disegni
- Frame vuoti a inizio tracking:
  - Spesso dovuti a warm-up del tracker; usa n_init=1 per ridurlo (se non già impostato)
- Sincronizzazione:
  - Se i frame comuni GT↔tracker sono 0, controlla la mappatura o gli offset di frame
- Performance:
  - Usa --device 0 (se previsto) o imposta device="cuda" nello script per GPU
- Versioni librerie:
  - Se ultralytics cambia API, verifica le chiamate a YOLO(model).predict
- Output rDetection/rTracked:
  - Cartelle per run alternativi/rapidi o di riferimento; condividono lo stesso formato dei file in detection/ e tracked/

---

## Comandi rapidi

- python detection.py
- python tracking.py
- python metrics.py
- python getVideoFrames.py
-