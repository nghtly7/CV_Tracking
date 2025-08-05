import os
import pandas as pd
import motmetrics as mm
import json
from glob import glob
from enum import Enum

# --- CONFIGURAZIONE: MODIFICA QUESTI PERCORSI ---
# Percorso alla cartella delle etichette (ground truth) di Roboflow per una singola angolazione
GROUND_TRUTH_LABELS_FOLDER = 'GroundTruthData/labels'

# Percorso al file JSON con i risultati del tracking
# (come quello salvato dall'esempio precedente)
TRACKING_RESULTS_FILE = '2d_tracking_results.json'

# Scegli l'angolazione che vuoi valutare
ANGLE_TO_EVALUATE= 'out13'
# Dimensioni dell'immagine (necessario per convertire le coordinate YOLO normalizzate)
IMAGE_WIDTH = 3840  # Ad esempio
IMAGE_HEIGHT = 2160 # Ad esempio

class RB_CLASSES(Enum):
        Ball = 0
        Red_0 = 1
        Red_11 = 2
        Red_12 = 3
        Red_16 = 4
        Red_2 = 5
        Refree_F = 6
        Refree_M = 7
        White_13 = 8
        White_16 = 9
        White_25 = 10
        White_27 = 11
        White_34 = 12

# --- FUNZIONI DI SUPPORTO PER CARICARE I DATI ---

def load_roboflow_yolo_annotations(labels_folder, image_width, image_height):
    """
    Carica le annotazioni YOLO di Roboflow e le converte in un DataFrame.
    Assegna un ID temporaneo per ogni oggetto in ogni frame.
    """
    

    #id delle classi di roboflow
    from enum import Enum

    
    
    gt_data = []
    yolo_files = [f for f in glob(os.path.join(labels_folder, '*.txt')) if ANGLE_TO_EVALUATE in os.path.basename(f)]
    
    for file_path in yolo_files:
        
        # Estrai l'ID del frame da 'frame_00001.txt'
        try:
            frame_id = int(file_path.split('_')[2])*5
        except (ValueError, IndexError):
            print(f"Warning: Could not parse frame_id from filename {file_path}. Skipping.")
            continue
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            try:
                parts = line.strip().split(' ')
                # Ignora le righe vuote o malformate
                if len(parts) < 5:
                    continue
                class_id, x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[:5])
                
                # Converti le coordinate normalizzate in pixel [x, y, w, h]
                x_center = x_center_norm * image_width
                y_center = y_center_norm * image_height
                width = width_norm * image_width
                height = height_norm * image_height
                
                x = x_center - width / 2
                y = y_center - height / 2
                
                # Assegna un ID temporaneo univoco all'interno del frame
                gt_data.append([frame_id, int(class_id), x, y, width, height])
            except (ValueError, IndexError):
                print(f"Warning: Malformed line in {file_path}. Skipping.")
                continue
            
    return pd.DataFrame(gt_data, columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height'])

def load_tracking_results_from_json(json_file, angle_to_evaluate):
    """
    Carica i risultati del tracking dal file JSON prodotto dallo script precedente.
    """
    with open(json_file, 'r') as f:
        all_results = json.load(f)

    tracking_data = []
    if angle_to_evaluate in all_results:
        for frame_data in all_results[angle_to_evaluate]:
            frame_id = frame_data['frame_id']
            for obj in frame_data['objects']:
                bbox = obj['bbox_xyxy']
                x, y, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                width = x2 - x
                height = y2 - y
                
                
                # 'obj_id' viene dall'ID univoco del tracker
                # Fissa: gestisci il caso in cui la classe non sia presente in RB_CLASSES
                class_name = obj['class']
                if class_name in RB_CLASSES.__members__:
                    class_id = RB_CLASSES[class_name].value
                    
                else:
                    print(f"Attenzione: classe '{class_name}' non trovata in RB_CLASSES. Salto l'oggetto.")
                    continue
                tracking_data.append([frame_id, class_id, x, y, width, height])

    return pd.DataFrame(tracking_data, columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height'])


# --- LOGICA DI VALUTAZIONE PRINCIPALE ---
if __name__ == '__main__':
    print(f"--- Valutazione del tracking per l'angolazione: {ANGLE_TO_EVALUATE} ---")

    # 1. Carica i dati del ground truth (le tue annotazioni di Roboflow)
    gt = load_roboflow_yolo_annotations(
        labels_folder=GROUND_TRUTH_LABELS_FOLDER,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT
    )
    # Converti i DataFrame in formati adatti per l'accumulatore, indicizzando per FrameId e Id
    gt_frames = gt.set_index(['FrameId', 'Id'])
    print(f"Caricate {len(gt)} annotazioni di ground truth su {len(gt.FrameId.unique())} frame.")
    
    # 2. Carica i risultati del tracking
    ts = load_tracking_results_from_json(TRACKING_RESULTS_FILE, ANGLE_TO_EVALUATE)
    ts_frames = ts.set_index(['FrameId', 'Id'])
    print(f"Caricati {len(ts)} risultati di tracking su {len(ts.FrameId.unique())} frame.")

    # 3. Allineamento e calcolo delle metriche
    # Crea un accumulatore di metriche
    acc = mm.MOTAccumulator(auto_id=True)

    # *** PUNTO CRUCIALE: Iteriamo SOLO sui frame che hanno un ground truth ***
    all_gt_frames_ids = gt.FrameId.unique()
    print(f"Valutazione in corso su {len(all_gt_frames_ids)} frame (quelli annotati).")
    
    for frame_id in all_gt_frames_ids:
        # Prendi i dati del frame corrente
        gt_frame = gt_frames.loc[[frame_id]] if frame_id in gt_frames.index else pd.DataFrame()
        ts_frame = ts_frames.loc[[frame_id]] if frame_id in ts_frames.index else pd.DataFrame()
        
        # Se non ci sono tracking per questo frame, passa al prossimo
        if ts_frame.empty:
            continue

        # Calcola la matrice delle distanze IoU (Intersection over Union) tra ground truth e tracking
        # La soglia IoU (max_iou) definisce quando un accoppiamento è valido
        distances = mm.distances.iou_matrix(ts_frame[['X', 'Y', 'Width', 'Height']],gt_frame[['X', 'Y', 'Width', 'Height']],max_iou=0.5) # Soglia di IoU: 0.5 è un valore comune

        # Aggiorna l'accumulatore
        acc.update(
            gt_frame.index.get_level_values('Id'),
            ts_frame.index.get_level_values('Id'),
            distances
        )
    # 4. Calcola il riassunto delle metriche
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='overall')
    

    # 5. Stampa i risultati
    print("\n--- RISULTATI DELLE METRICHE DI TRACKING ---")
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))