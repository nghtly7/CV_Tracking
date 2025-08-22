# 3D Tracking – Triangolazione, Tracking, Viewer, Metriche, Export Unreal

Pipeline completa per il tracking 3D in ambito sportivo: dalle osservazioni 2D multi-camera fino a metriche su piano campo, visualizzazione e export per Unreal Engine.

Contenuti:
- Struttura cartelle e prerequisiti
- Dati e convenzioni
- Pipeline end-to-end
- Configurazione per script
- Esecuzione (Quickstart)
- Formati di input/output
- Suggerimenti e troubleshooting

---

## Struttura cartelle e prerequisiti

Struttura tipica della cartella 3D_tracking:
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
  - unreal_tracks.csv, unreal_frames.jsonl (output export Unreal)

Requisiti (consigliato Python 3.10+):
- numpy, pandas, matplotlib, scipy, opencv-python (per triangolazione/utility)

Installazione rapida (PowerShell):
- pip install numpy pandas matplotlib scipy opencv-python

---

## Dati e convenzioni

- Spazio 3D:
  - Unità: metri
  - Z up; XY sul piano campo
- Classi:
  - Canoniche: player, referee, ball
  - Mapping supportato anche da ID: 1→player, 2→referee, 0→ball
- CSV di predizioni 3D (tracks3d.csv):
  - Colonne tipiche: t, track_id, class, x, y, z, vx, vy, vz, meas_err_px
  - Sono accettati alias per colonne (t|frame, x|x_m, …) negli script che lo prevedono

---

## Pipeline end-to-end

1) Triangolazione (triangulation.py)
- Carica calibrazioni (K, R, t) e osservazioni 2D rettificate
- Matching epipolare e clustering multi-vista
- Triangolazione (DLT) + refine nonlineare (LM) + stima covarianza
- Filtri: errore di riproiezione, n. viste minime
- Output per frame in triangulations/

2) Tracking 3D (3D_tracker.py)
- Filtra e deduplica triangolazioni
- Modello CV per player/referee; modello balistico per ball
- Associazione con distanza di Mahalanobis e gating chi^2
- Gestione conferma/assenza/terminazione tracce
- Output snapshot: tracks3d/tracks3d.csv e stats.json

3) Visualizzatore 3D (displayData.py)
- Riproduzione frame-by-frame con controlli tastiera
- Disegno campo (bounding box dati o dimensioni preimpostate)
- Frecce direzione (yaw) opzionali per player/referee

4) Metriche 3D (3dMetrics.py)
- Proiezione GT su piano campo via omografie
- Allineamento temporale GT→pred (FRAME_SCALE/OFFSET)
- Dedup GT e post-processing pred opzionali
- Matching metrico e calcolo metriche (detection/posizione, tracking opzionale)

5) Export per Unreal Engine (export_unreal_engine.py)
- Conversione metri→centimetri, rotazione/offset mondo, stima yaw
- CSV per DataTable/Blueprint/Sequencer + JSONL per frame

---

## Configurazione per script

Triangolazione (triangulation.py)
- Percorsi calibrazioni e osservazioni
- Soglie per errore di riproiezione, min views

Tracking 3D (3D_tracker.py)
- Parametri modello (rumori, gravità), gating chi^2
- Soglie dedup e gestione conferme/miss
- FPS

Viewer 3D (displayData.py)
- CSV_PATH: percorso al CSV (metri, Z up)
- FPS: fps per barra tempo
- FIELD_SIZE: None oppure (LUNGHEZZA, LARGHEZZA) in metri

Metriche 3D (3dMetrics.py)
- TRACKS3D_CSV, COCO_GT_PATH, CAMERA_DATA
- FRAME_SCALE, FRAME_OFFSET
- Gating e soglie per matching e dedup GT
- Abilitazione post-processing (stitch/speed/smooth/NMS) e tracking metrics

Export Unreal (export_unreal_engine.py)
- INPUT_CSV: CSV d’ingresso (metri)
- OUT_DIR: cartella output (es. 3D_tracking/unreal)
- FPS_TRACKS: fps delle tracce (per time_s)
- CM_PER_M: 100.0 (UE usa cm)
- WORLD_ROT_DEG: rotazione antioraria piano XY (gradi)
- WORLD_OFFSET_M: (x, y, z) in metri
- REBASE_TIME: se True, time_s parte da min(t)
- WRITE_JSONL, SORT_OUTPUT

---

## Esecuzione (Quickstart)

PowerShell (Windows):
- cd c:\Users\nicol\Desktop\CV_Tracking\3D_tracking

Triangolazione:
- python triangulation.py

Tracking 3D:
- python 3D_tracker.py

Visualizzatore:
- python displayData.py
- Tasti: SPACE (play/pausa), ←/→ (frame), ↑/↓ (velocità), Q (esci)

Metriche:
- python 3dMetrics.py

Export Unreal:
- python export_unreal_engine.py
- Output in: 3D_tracking/unreal

---

## Formati di input/output

Predizioni 3D (tracks3d/tracks3d.csv)
- t (int), track_id (int), class (str|int), x, y, z (float, metri)
- Opzionali: vx, vy, vz, meas_err_px

Export Unreal
- unreal_tracks.csv (centimetri, Z up)
  - row_name: "{class}_{track_id}_{frame}"
  - track_id, class, frame, time_s, x_cm, y_cm, z_cm, yaw_deg
- unreal_frames.jsonl (opzionale, 1 riga per frame)
  - Per frame t: { "frame": t, "time_s": ..., "objects": [ { "id", "class", "x", "y", "z", "yaw" }, ... ] }

Viewer 3D
- Legge tracks3d.csv (alias colonne supportati)
- Disegna 3 scatter per classi + quiver per direzione

Metriche (3dMetrics.py)
- metrics_summary.json con metrica detection (per soglie), posizione (MAE/RMSE/percentili), tracking opzionale (CLEAR-MOT/IDF1)

---

## Suggerimenti e troubleshooting

- Colonne/alias:
  - Gli script accettano alias per t|frame, track_id|id, class|label|category, x|x_m, y|y_m, z|z_m
- Classi:
  - Accetta sia nomi che ID; mapping interno normalizza a player/referee/ball
- Yaw:
  - Calcolato dalla direzione nel piano XY in world space; se oggetto fermo → yaw=0
- Unreal:
  - Impostare WORLD_ROT_DEG/WORLD_OFFSET_M per allineare coordinate a livello mappa UE
  - UE usa centimetri; output già convertito
- Visualizzatore:
  - Se non si specifica FIELD_SIZE, usa il bounding box XY dei dati
- Allineamento metriche:
  - FRAME_SCALE/OFFSET devono riflettere il rapporto FPS GT↔pred (es. GT 5 fps vs pred 25fps)
- Troubleshooting:
  - Errori comuni includono:
    - FRAME_SCALE/OFFSET errati → allineamento temporale sbagliato
    - Calibrazioni incoerenti → triangolazione/associazione errata
    - Parametri di gating troppo stretti → tracce mancanti o frammentate
    - Alias colonne non riconosciuti → errore lettura CSV
  - Controllare i log di output per messaggi di errore o avviso
  - Verificare visivamente i risultati con il visualizzatore 3D
  - Eseguire il debug passo-passo per isolare problemi in specifici script o funzioni

---

## Note di progettazione

- Normalizzazione ID camera: outN → cam_N (regex/normalizer).
- Punto di confronto GT:
  - Player/Referee: bottom-center bbox (punto a terra).
  - Ball: centro bbox (può essere in aria).
- Stabilità/performance:
  - Cache di H^(-1) per camera, soglie conservative per stitching e NMS.
- Alcune parti possono richiedere completamento/rifinitura del codice dove presenti