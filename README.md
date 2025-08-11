# CV_Tracking

Questo progetto esegue Detection, Tracking e Valutazione delle metriche su video di calcio, con pipeline basata su YOLO e Deep SORT.

- Detection: [detection.py](detection.py)
- Tracking: [tracking.py](tracking.py)
- Valutazione (pipeline COCO): [metrics.py](metrics.py)
- Valutazione (pipeline MOT basata su YOLO labels): [metricsEvaluation.py](metricsEvaluation.py)

Cartelle di supporto:
- GroundTruth COCO: GroundTruthData/train/_annotations.coco.json
- Output Detection: detection/
- Output Tracking: tracked/
- Video grezzi: raw_video/
- Strumenti dataset: datasetManipulation/
- Materiale di supporto: support_material/

Requisiti principali:
- Python 3.10+
- pacchetti: ultralytics, opencv-python, deep_sort_realtime, numpy, matplotlib, motmetrics, pandas

Installazione rapida (PowerShell):
- python -m venv .venv
- .\.venv\Scripts\Activate.ps1
- pip install ultralytics opencv-python deep_sort_realtime numpy matplotlib motmetrics pandas

Struttura delle cartelle (essenziale)
- raw_video/
  - out2.mp4
  - out4.mp4
  - out13.mp4
- detection/
  - outX_detections.json (output di detection.py)
  - outX_annotated.mp4 (video annotato)
- tracked/
  - outX_tracks.json (output di tracking.py)
  - outX_tracked.mp4 (video tracciato)
- GroundTruthData/
  - train/
    - _annotations.coco.json (COCO) — usato da metrics.py
  - labels/ (YOLO txt) — usato da metricsEvaluation.py
- scripts principali:
  - detection.py
  - tracking.py
  - metrics.py
  - metricsEvaluation.py

Pipeline 1: Detection → Tracking → Metriche (COCO)

1) Detection
- Edita il video da processare in [detection.py](detection.py):
  - video_path = "raw_video/out4.mp4"
  - model_path = "best.pt" (modello YOLO addestrato)
- Output:
  - detection/out4_detections.json
  - detection/out4_annotated.mp4
- Formato output detections: per frame, lista “detections” con:
  - [x, y, w, h, conf, class_id] in pixel (XYWH, assoluti)
- Esecuzione:
  - python detection.py

2) Tracking
- Edita il video in [tracking.py](tracking.py) (coerente con detection):
  - video_path = "raw_video/out4.mp4"
  - Carica detection da detection/{video_name}_detections.json automaticamente
- Tracker: Deep SORT con n_init=1 (conferma dopo 1 hit). Salva solo tracce aggiornate e confermate nel frame (filtri per evitare confidenza/classe nulle).
- Output:
  - tracked/out4_tracks.json
  - tracked/out4_tracked.mp4
- Formato output tracking per frame:
  - {
      "frame": frame_id,
      "tracks": [
        { "id": int, "bbox": [x1,y1,x2,y2], "confidence": float, "class_id": int },
        ...
      ]
    }
- Esecuzione:
  - python tracking.py

3) Metriche (COCO)
- Configura il video target in [metrics.py](metrics.py):
  - video = "out4"  (usato per filtrare le immagini COCO e allineare i frame)
- Ground Truth: [GroundTruthData/train/_annotations.coco.json](GroundTruthData/train/_annotations.coco.json)
  - Le immagini devono avere nomi contenenti video e indice frame (pattern es. out4_frame_0001_...)
  - La funzione load_coco_annotations filtra per ‘video’ e converte i bbox [x,y,w,h] → [x1,y1,x2,y2]
  - Mappatura dei frame: i frame GT sono sottocampionati; GT 1→ tracker 3, GT 2→ tracker 8, GT 3→ tracker 13 (cioè tracker = -2 + 5*GT)
- Il tracker viene caricato da tracked/{video}_tracks.json
- Metriche calcolate in evaluate_tracking:
  - MOTA, MOTP (IoU medio), Precision, Recall, F1, ID_Switches, Fragments, FP, FN, Matches
- Note importanti:
  - ID switch: usa l’ID “id” del tracker come obj_id
  - Le identità GT sono costruite via build_gt_tracks (linking per IoU tra frame GT) per contare IDSW in modo robusto
  - Fragments: conteggiati come numero di segmenti matched per GT track meno 1
- Esecuzione:
  - python metrics.py
  - Genera anche grafici PNG di metriche e errori

Pipeline 2 (alternativa): Valutazione con motmetrics (YOLO labels)
- Usa [metricsEvaluation.py](metricsEvaluation.py) se vuoi valutare contro etichette YOLO (txt in GroundTruthData/labels)
- Da configurare in testa al file:
  - GROUND_TRUTH_LABELS_FOLDER
  - TRACKING_RESULTS_FILE (es. 2d_tracking_results.json prodotto da yolo_byteTracking.py)
  - ANGLE_TO_EVALUATE (es. 'out13')
  - IMAGE_WIDTH, IMAGE_HEIGHT
- Esecuzione:
  - python metricsEvaluation.py
- Output: riepilogo standard MOTChallenge via motmetrics

Cosa cambiare per usare un video diverso

Caso A — Pipeline Detection/Tracking/Metriche (COCO):
- detection.py:
  - video_path = "raw_video/out13.mp4"
  - model_path se necessario
- tracking.py:
  - video_path = "raw_video/out13.mp4"
  - Il resto viene derivato da video_name (out13)
- metrics.py:
  - video = "out13"
  - Assicurati che il COCO [GroundTruthData/train/_annotations.coco.json](GroundTruthData/train/_annotations.coco.json) contenga immagini di out13 con file_name coerente col parser (pattern primario: r'([^_]+)_frame_(\d+)_').
    - Se la tua naming convention è diversa, modifica i pattern regex in load_coco_annotations.
- Verifica che esistano:
  - detection/out13_detections.json (dal passo 1)
  - tracked/out13_tracks.json (dal passo 2)

Caso B — Pipeline con YOLO labels (motmetrics):
- metricsEvaluation.py:
  - ANGLE_TO_EVALUATE = 'out4' (o altro)
  - GROUND_TRUTH_LABELS_FOLDER = 'GroundTruthData/labels'
  - TRACKING_RESULTS_FILE = '2d_tracking_results.json' (da [yolo_byteTracking.py](yolo_byteTracking.py))
  - IMAGE_WIDTH/HEIGHT coerenti al video
- Opzionale: Estrai i frame se lavori per immagini con [getVideoFrames.py](getVideoFrames.py)

Note utili e debug
- Formati bbox:
  - Detection: [x,y,w,h,conf,class_id] (XYWH, pixel)
  - Tracking: bbox in [x1,y1,x2,y2] (XYXY, pixel)
  - COCO GT: bbox in [x,y,w,h] → convertiti a XYXY in load_coco_annotations
- Perché i primi frame del tracking possono essere vuoti:
  - Il tracker salva solo tracce confermate e aggiornate nel frame (track.is_confirmed() e time_since_update==0)
  - Con n_init=1 si riduce il warm-up
- Verifica IoU:
  - In [metrics.py](metrics.py) sono disponibili quick_iou_self_test() e debug_ious(...) per controlli rapidi
- Conversione classi GT:
  - gt_class_conversion: 1→0, 6/7→2, altrimenti→1
- Allineamento temporale GT ↔ Tracker:
  - GT n → Tracker (-2 + 5*n): 1→3, 2→8, 3→13, ...
  - La funzione load_coco_annotations calcola l’indice “actual_frame_idx” coerente per il confronto

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