# 3D Tracking – Triangolazione, Tracking e Valutazione

Questo modulo realizza una pipeline completa per il tracking 3D di oggetti in un contesto sportivo, a partire da osservazioni 2D multi-camera fino al calcolo delle metriche di valutazione su piano campo.

Componenti principali:
- Triangolazione 3D delle osservazioni 2D multi-vista
- Costruzione di tracce 3D con filtri/associazioni robusti
- Valutazione metrica (detection/posizione e opzionale tracking) vs Ground Truth COCO rettificato

Sezioni:
- Struttura del codice
- Dati in ingresso e convenzioni
- Pipeline end-to-end
- Configurazione e parametri chiave
- Esecuzione (Quickstart)
- Output attesi
- Suggerimenti e troubleshooting

---

## Struttura del codice

- triangulation.py
  - Carica calibrazioni (K, R, t) e osservazioni 2D per camera.
  - Effettua matching epipolare (distanza di Sampson + costi compositi), clustering multi-vista.
  - Triangola (DLT), rifinisce con Levenberg–Marquardt, stima covarianza.
  - Filtra per errore di riproiezione / n. viste e salva per frame:
    - triangulations/associations_*.json e/o triangulated_*.json.
- 3D_tracker.py
  - Carica i punti triangolati (triangulated_*.json).
  - Filtra qualità e deduplica per frame/classe (fusione gaussiana con soglie euclidee + test di Mahalanobis).
  - Tracker 3D:
    - Modello CV (costante velocità) per player/referee.
    - Modello balistico per palla (gravità, accelerazione più rumorosa).
  - Associazione per distanza di Mahalanobis con gating chi^2.
  - Gestione conferma/assenza/terminazione tracce.
  - Esporta snapshot: tracks3d/tracks3d.csv e tracks3d/stats.json.
- 3dMetrics.py
  - Carica predizioni 3D (CSV) e GT COCO rettificato.
  - Carica calibrazioni e costruisce omografie per proiettare GT a terra (piano z=0).
  - Allinea temporalmente GT → timeline predizioni (FRAME_SCALE/OFFSET).
  - Proietta GT (pixel → campo) e deduplica cross-camera per classe.
  - Post-processing pred (opzionale): stitching, filtro lunghezza minima, filtro velocità “two-strike”, smoothing zero-lag, NMS metrico framewise.
  - Matching GT↔Pred per-classe (Hungarian con gate metrico).
  - Metriche:
    - Detection: Precision/Recall/F1 a varie soglie (metri).
    - Posizione: MAE/RMSE/percentili delle distanze sul piano campo.
    - Tracking (opzionale): CLEAR-MOT e IDF1 sul piano campo.
  - Stampa tabelle e salva metrics_summary.json.

Nota: Alcune funzioni nei file possono essere placeholder/in completamento; il flusso descritto rappresenta il comportamento previsto.

---

## Dati in ingresso e convenzioni

- Calibrazioni: support_material/3D_tracking_material/camera_data/cam_*/calib/camera_calib.json
  - Intrinseche K, rotazione R, traslazione t (unità m; conversione mm→m opzionale).
  - Se disponibili intrinseche rettificate (K_rect/Knew), vengono preferite.
- Osservazioni 2D: 2D_tracking/rTracked (outX_tracks.json o simili)
  - Coordinate in immagini rettificate (coerenti con COCO rettificato).
- Ground Truth (GT): GroundTruthData/train/_annotations_rectified.coco.json
  - Bbox su immagini rettificate. Per player/referee si usa il bottom-center (punto a terra); per ball il centro bbox.
  - Mappatura classi normalizzata: 0=ball, 1=player, 2=referee.
- Predizioni 3D: 3D_tracking/tracks3d/tracks3d.csv
  - Colonne tipiche: t, track_id, class, x, y, z (metri, piano campo su xy).

Convenzioni:
- Camera ID normalizzati: outN → cam_N.
- Piano campo: proiezione tramite omografia H = K [r1 r2 t] assumendo z=0.
- Allineamento temporale GT → pred: t_pred = FRAME_OFFSET + FRAME_SCALE * f_gt (es. 5x per 25fps vs 5fps).

---

## Pipeline end-to-end

1) Triangolazione (triangulation.py)
- Carica K, R, t per ciascuna camera; opzionale rettifica intrinseche.
- Matching epipolare tra coppie di camere con distanza di Sampson e costo combinato (classe, scala bbox, ecc. se previsto).
- Clustering multi-vista delle corrispondenze.
- Triangolazione DLT → refine nonlineare (LM) minimizzando l’errore di riproiezione.
- Stima covarianza del punto 3D.
- Filtri: errore di riproiezione, n. viste minime; salva JSON per frame.

2) Tracking 3D (3D_tracker.py)
- Carica triangulated_*.json e applica filtri qualità + dedup per classe.
- Tracking:
  - Player/referee: Kalman CV con rumore di accelerazione moderato.
  - Ball: modello balistico con gravità e rumore accelerazione maggiore.
- Associazione: distanza di Mahalanobis, gating chi^2 (dof=3).
- Gestione tracce: conferma dopo N hit, persistenza a miss (MAX_MISSES_*), terminazione.
- Output: tracks3d.csv con snapshot per frame + stats.json.

3) Valutazione (3dMetrics.py)
- Carica predizioni e GT, ricava dimensioni immagine per camera dal COCO.
- Calibrazioni + omografie H per proiettare GT da pixel a metri (xy campo).
- Allinea GT ai frame pred (FRAME_SCALE/OFFSET), filtra ai frame sovrapposti.
- Dedup GT cross-camera per frame e classe con raggio per-classe.
- Post-processing pred (opzionale):
  - Stitching di tracce vicine nel tempo/spazio coerenti in direzione/velocità, senza overlap.
  - Filtro di lunghezza minima.
  - Filtro velocità “two-strike” (due step consecutivi oltre soglia per rimuovere outlier).
  - Smoothing zero-lag (media mobile centrata) per ridurre jitter senza ritardo.
  - NMS metrico per frame/classe per ridurre duplicati vicini (raggio per-classe).
- Matching Hungarian per classe con gate metrico; calcolo:
  - Detection @ soglie in metri: TP/FP/FN → Precision/Recall/F1.
  - Posizione: distanze dei match → MAE/RMSE/percentili.
  - Tracking (opzionale): CLEAR-MOT, IDF1 (se abilitato e implementato).
- Stampa riepilogo e salva metrics_summary.json.

---

## Configurazione e parametri chiave

- triangulation.py
  - CALIB_ROOT, OBSERVATIONS_PATH, OUT_DIR
  - Filtri: MIN_VIEWS, soglie errore riproiezione
- 3D_tracker.py
  - MAX_REPROJ_PX_*, MIN_VIEWS, DEDUP_THRESH_* (m)
  - Gating: CHI2_GATE_3D, CHI2_MERGE_3D
  - Modelli/rumori: ACCEL_NOISE_*, GRAVITY, FPS
  - Miss/confirm: MAX_MISSES_*, MIN_HITS_CONFIRM_*
- 3dMetrics.py
  - FRAME_SCALE, FRAME_OFFSET
  - MATCH_GATE_M, MATCH_GATE_BY_CLASS
  - GT_MERGE_RADIUS_M, MERGE_RADIUS_BY_CLASS
  - Post-processing: ENABLE_* (stitch/speed/smooth/NMS), relativi parametri
  - ENABLE_TRACKING_METRICS, LINK_GATE_BY_CLASS, LINK_MAX_FRAME_GAP
  - Percorsi: TRACKS3D_CSV, COCO_GT_PATH, CAMERA_DATA, OUT_JSON

Suggerimento: versionare i parametri con il JSON di output per tracciabilità.

---

## Esecuzione (Quickstart)

Prerequisiti:
- Python 3.10+ (consigliato), Windows
- pacchetti: numpy, pandas, opencv-python, scipy

Installazione pacchetti:
```powershell
pip install numpy pandas opencv-python scipy
```

1) Triangolazione
```powershell
cd c:\Users\nicol\Desktop\CV_Tracking\3D_tracking
python triangulation.py
```

2) Tracking 3D
```powershell
python 3D_tracker.py
```

3) Valutazione
```powershell
python 3dMetrics.py
```

Note:
- Verificare FRAME_SCALE/OFFSET in 3dMetrics.py in base agli FPS effettivi GT/pred.
- Assicurarsi che le calibrazioni contengano K, R, t coerenti con le immagini rettificate del GT.

---

## Output attesi

- triangulations/
  - associations_#.json, triangulated_#.json: cluster, triangolazioni, errori di riproiezione, covarianze.
- tracks3d/
  - tracks3d.csv: t, track_id, class, x, y, z, … (metri)
  - stats.json: statistiche tracce (durata, miss, conferme).
- metrics_summary.json
  - Parametri usati, metriche detection per soglia, metriche di posizione overall e per classe, frames valutati.

---

## Suggerimenti e troubleshooting

- Calibrazioni (unità t):
  - Se t è in mm, abilitare ASSUME_T_MM=True (o conversione automatica in base alla norma).
- Omografie H e proiezione:
  - Se l’inversione H fallisce/instabile, verificare K rettificata e dimensioni immagine; usare DEFAULT_IMG_SIZE come fallback.
- Allineamento temporale:
  - Se “No aligned frames”, controllare FRAME_SCALE/OFFSET e che i frame pred siano presenti in tracks3d.csv.
- Gating Hungarian:
  - Se F1 è troppo basso nonostante buona qualità, aumentare MATCH_GATE_BY_CLASS o le soglie PREC_THRESH_M.
- Deduplicazione GT:
  - Adattare MERGE_RADIUS_BY_CLASS per scenari con soggetti densi/affollati.
- Post-processing:
  - Disattivare temporaneamente ENABLE_* per isolare l’impatto di ciascun filtro/stage.

---

## Note di progettazione

- Normalizzazione ID camera: outN → cam_N (regex/normalizer).
- Punto di confronto GT:
  - Player/Referee: bottom-center bbox (punto a terra).
  - Ball: centro bbox (può essere in aria).
- Stabilità/performance:
  - Cache di H^(-1) per camera, soglie conservative per stitching e NMS.
- Alcune parti possono richiedere completamento/rifinitura del codice dove presenti