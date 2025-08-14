# CV_Tracking

Questo progetto esegue Detection, Tracking e Valutazione delle metriche su video di calcio, con pipeline basata su YOLO e Deep SORT.

- Detection: [detection.py](detection.py)
- Tracking: [tracking.py](tracking.py)
- Valutazione (pipeline COCO): [metrics.py](metrics.py)
- Valutazione (pipeline MOT basata su YOLO labels): [metricsEvaluation.py](metricsEvaluation.py)

Requisiti principali:
- Python 3.10+
- pacchetti: ultralytics, opencv-python, deep_sort_realtime, numpy, matplotlib, motmetrics, pandas

Installazione rapida (PowerShell):
- python -m venv .venv
- .\.venv\Scripts\Activate.ps1
- pip install ultralytics opencv-python deep_sort_realtime numpy matplotlib motmetrics pandas

Struttura gerarchica cartelle
- raw_video/
  - out2.mp4
  - out4.mp4
  - out13.mp4
- detection/
  - outX_detections.json         (output di [detection.py](detection.py))
  - outX_annotated.mp4           (video con bbox di detection)
- tracked/
  - outX_tracks.json             (output di [tracking.py](tracking.py))
  - outX_tracked.mp4             (video con bbox e ID di tracking)
- GroundTruthData/
  - train/
    - _annotations.coco.json     (COCO GT per [metrics.py](metrics.py))
  - labels/                      (annotazioni YOLO txt per [metricsEvaluation.py](metricsEvaluation.py))
- datasetManipulation/
  - labelCorrection.py
  - datasetStats.py
- support_material/
- script principali a radice repo:
  - detection.py, tracking.py, metrics.py, metricsEvaluation.py, README.md

Formati dati
- Detection (detection/outX_detections.json)
  - Per frame: "detections": [x, y, w, h, conf, class_id] (XYWH, pixel)
- Tracking (tracked/outX_tracks.json)
  - Per frame:
    {
      "frame": int,
      "tracks": [
        { "id": int, "bbox": [x1,y1,x2,y2], "confidence": float, "class_id": int }
      ]
    }
- Ground Truth COCO (GroundTruthData/train/_annotations.coco.json)
  - bbox in [x, y, w, h] (convertiti a XYXY in [metrics.py](metrics.py))

Pipeline 1: Detection → Tracking → Metriche (COCO)

1) Detection
- Configura in [detection.py](detection.py):
  - video_path = "raw_video/out4.mp4"
  - model_path = "best.pt"
- Output:
  - detection/out4_detections.json
  - detection/out4_annotated.mp4
- Esecuzione:
  - python detection.py

2) Tracking
- Configura in [tracking.py](tracking.py):
  - video_path = "raw_video/out4.mp4"
  - Carica automaticamente detection/dello stesso video
- Caratteristiche:
  - Deep SORT con n_init=1
  - Salva solo tracce aggiornate nel frame e confermate (niente valori nulli)
  - Colori bbox: classe 0=verde, classe 1=azzurro (cyan), classe 2=rosso
- Output:
  - tracked/out4_tracks.json
  - tracked/out4_tracked.mp4
- Esecuzione:
  - python tracking.py

3) Metriche (COCO)
- Configura il video target in [metrics.py](metrics.py):
  - video = "out4"
- Ground Truth:
  - [GroundTruthData/train/_annotations.coco.json](GroundTruthData/train/_annotations.coco.json)
  - La funzione load_coco_annotations filtra per ‘video’ e converte bbox a XYXY
  - Mappatura temporale: GT è a 5 fps, tracking a 25 fps
    - GT n → tracker (-2 + 5*n): 1→3, 2→8, 3→13, ...
- Il tracker viene caricato da tracked/{video}_tracks.json
- Metriche calcolate in evaluate_tracking:
  - MOTA, MOTP (IoU medio), Precision, Recall, F1, ID_Switches, Fragments, FP, FN, Matches
- Esecuzione:
  - python metrics.py

Pipeline 2 (alternativa): Valutazione con motmetrics (YOLO labels)
- Usa [metricsEvaluation.py](metricsEvaluation.py) con etichette YOLO (txt)
- Config da impostare in testa al file:
  - GROUND_TRUTH_LABELS_FOLDER = 'GroundTruthData/labels'
  - TRACKING_RESULTS_FILE = '2d_tracking_results.json'
  - ANGLE_TO_EVALUATE = 'out4'
  - IMAGE_WIDTH, IMAGE_HEIGHT
- Esecuzione:
  - python metricsEvaluation.py
- Output:
  - Riepilogo standard MOTChallenge con motmetrics

Come usare un video diverso

Caso A — Pipeline COCO (detection/tracking/metriche):
- [detection.py](detection.py):
  - video_path = "raw_video/out13.mp4"
- [tracking.py](tracking.py):
  - video_path = "raw_video/out13.mp4"
- [metrics.py](metrics.py):
  - video = "out13"
  - Assicurati che il COCO contenga immagini per quel video con un filename parsabile (pattern primario r'([^_]+)_frame_(\d+)_')
- Verifica file:
  - detection/out13_detections.json
  - tracked/out13_tracks.json

Caso B — Pipeline YOLO labels (motmetrics):
- [metricsEvaluation.py](metricsEvaluation.py):
  - ANGLE_TO_EVALUATE = 'out13'
  - GROUND_TRUTH_LABELS_FOLDER, TRACKING_RESULTS_FILE, IMAGE_WIDTH/HEIGHT coerenti

Note utili e debug
- Formati bbox:
  - Detection: [x,y,w,h,conf,class_id] (XYWH, pixel)
  - Tracking: [x1,y1,x2,y2] (XYXY, pixel)
  - COCO GT: [x,y,w,h] → XYXY in load_coco_annotations
- Perché i primi frame del tracking possono risultare vuoti:
  - Salviamo solo tracce aggiornate e confermate nel frame (track.is_confirmed() e time_since_update==0). Con n_init=1 si riduce il warm-up
- IoU:
  - In [metrics.py](metrics.py) sono disponibili quick_iou_self_test() e debug_ious(...) per controlli rapidi
- Conversione classi GT:
  - gt_class_conversion: 1→0, 6/7→2, altrimenti→1
- Allineamento temporale GT ↔ Tracker:
  - GT n → tracker (-2 + 5*n): 1→3, 2→8, 3→13, ...
- ID Switch e Fragments:
  - ID switch conteggiati usando ID tracker "id" e identità GT costruite via linking IoU
  - Fragments = numero segmenti di match per traccia GT − 1

Modifiche recenti (changelog)
- [tracking.py](tracking.py)
  - Salvataggio solo di tracce aggiornate e confermate nel frame (evita confidenze/classi nulle)
  - Colori bbox fissi: 0=verde, 1=azzurro, 2=rosso
  - Fix disegno etichetta (rettangolo sfondo con pt2 corretto)
  - Output JSON per track: { "id", "bbox" [x1,y1,x2,y2], "confidence", "class_id" }
- [metrics.py](metrics.py)
  - load_tracker_results legge il campo "id" del tracker come obj_id
  - load_coco_annotations filtra per variabile video e mappa frame GT → tracker con formula (-2 + 5*n)
  - gt_class_conversion: 1→0, 6/7→2, altrimenti→1
  - build_gt_tracks: costruzione identità GT consistenti nel tempo via linking IoU
  - evaluate_tracking:
    - ID Switch senza vincolo di frame consecutivi (adatto a GT sottocampionato)
    - Fragments calcolati come segmenti di match per GT − 1
  - Utility: quick_iou_self_test e debug_ious per validare IoU
- [README.md](README.md)
  - Aggiornata gerarchia cartelle e istruzioni per cambiare video
  - Chiarito formato dei file di detection/tracking e mappatura dei frame

Comandi rapidi (PowerShell Windows)
- python detection.py
- python tracking.py
- python metrics.py

File principali da editare per cambiare video
- [detection.py](detection.py): video_path, model_path
- [tracking.py](tracking.py): video_path (detection_json è derivato)
- [metrics.py](metrics.py): video = "outX"
- [metricsEvaluation.py](metricsEvaluation.py): ANGLE_TO_EVALUATE, percorsi e dimensioni

Suggerimenti
- Assicurati che il nome file nelle immagini COCO contenga il “video” (es. out4) e l’indice frame parsabile dai pattern regex usati
- Controlla che tracked/{video}_tracks.json contenga “frame” e “tracks” con campi “id”, “bbox”, “confidence”, “class_id”
- Se i frame comuni GT↔tracker sono 0, verifica:
  - la mappatura dei frame