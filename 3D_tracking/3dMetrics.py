import json
import csv
import numpy as np
import pandas as pd
from collections import defaultdict
import os
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import math
import warnings
warnings.filterwarnings('ignore')

# Configurazione
FRAME_SCALE = 5  # GT è campionato a 5 fps vs 25 fps del tracker
FRAME_OFFSET = 3  # GT frame 0 corrisponde a tracker frame 3
VIDEO_NAME = "out4"  # Configurabile

def gt_class_conversion(class_id):
    """
    Convert ground truth class IDs to simplified classes:
    - Class 1 -> Class 0 (ball)
    - Class 7,8 -> Class 2 (referee) 
    - All others -> Class 1 (player)
    """
    if class_id == 1:
        return 0  # ball
    elif class_id in [7, 8]:
        return 2  # referee
    else:
        return 1  # player

def load_predictions(csv_path: str) -> pd.DataFrame:
    """
    Carica i risultati del tracking 3D dal file CSV.
    
    Args:
        csv_path: Percorso al file tracks3d.csv
        
    Returns:
        DataFrame con le predizioni del tracker
    """
    print(f"Loading predictions from {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} tracking predictions")
        print(f"Frames range: {df['t'].min()} - {df['t'].max()}")
        print(f"Unique track IDs: {df['track_id'].nunique()}")
        print(f"Classes: {df['class'].unique()}")
        return df
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return pd.DataFrame()

def load_calibrations_and_H(camera_data_path: str) -> Dict[str, dict]:
    """
    Carica calibrazioni e calcola matrici di omografia per tutte le camere.
    
    Args:
        camera_data_path: Percorso alla cartella camera_data
        
    Returns:
        Dizionario {cam_id: {'K': K, 'R': R, 't': t, 'H': H}}
    """
    cams = {}
    
    for cam_dir in sorted(Path(camera_data_path).glob("cam_*")):
        calib_file = cam_dir / "calib" / "camera_calib.json"
        if not calib_file.exists():
            continue
            
        print(f"Loading calibration from {calib_file}")
        
        with open(calib_file, 'r') as f:
            calib_data = json.load(f)
        
        # Matrice intrinseca K
        if "K" in calib_data:
            K = np.array(calib_data["K"], dtype=float).reshape(3, 3)
        elif "mtx" in calib_data:
            K = np.array(calib_data["mtx"], dtype=float).reshape(3, 3)
        else:
            # Costruisci K da parametri individuali
            fx = calib_data.get("fx", 1000)
            fy = calib_data.get("fy", 1000)
            cx = calib_data.get("cx", 320)
            cy = calib_data.get("cy", 240)
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        
        # Matrice di rotazione R
        if "R" in calib_data:
            R = np.array(calib_data["R"], dtype=float).reshape(3, 3)
        else:
            # Converti da rvec a matrice di rotazione
            rvec_key = "rvecs" if "rvecs" in calib_data else "rvec"
            rvec = np.array(calib_data[rvec_key], dtype=float).reshape(3, 1)
            import cv2
            R, _ = cv2.Rodrigues(rvec)
        
        # Vettore di traslazione t
        if "t" in calib_data:
            t = np.array(calib_data["t"], dtype=float).reshape(3, 1)
        else:
            tvec_key = "tvecs" if "tvecs" in calib_data else "tvec"
            t = np.array(calib_data[tvec_key], dtype=float).reshape(3, 1)
        
        # Calcola matrice di omografia H = K * [r1 r2 t]
        # Per proiezione piano z=0 nel sistema mondo
        H = K @ np.column_stack([R[:, 0], R[:, 1], t.reshape(3)])
        
        cams[cam_dir.name] = {
            "K": K,
            "R": R,
            "t": t,
            "H": H
        }
    
    if not cams:
        print("WARNING: No camera calibrations found!")
        # Fallback: crea calibrazione identità
        cams["cam_default"] = {
            "K": np.eye(3),
            "R": np.eye(3),
            "t": np.zeros((3, 1)),
            "H": np.eye(3)
        }
    
    return cams

def img_to_field_xy(H: np.ndarray, u: float, v: float) -> Tuple[float, float]:
    """
    Converte coordinate pixel in coordinate campo usando omografia inversa.
    
    Args:
        H: Matrice di omografia 3x3
        u, v: Coordinate pixel
        
    Returns:
        Tuple (x, y) coordinate campo in metri
    """
    try:
        H_inv = np.linalg.inv(H)
        pixel_point = np.array([u, v, 1.0], dtype=float)
        field_point = H_inv @ pixel_point
        
        if abs(field_point[2]) > 1e-10:
            x = field_point[0] / field_point[2]
            y = field_point[1] / field_point[2]
        else:
            x = y = 0.0
            
        return float(x), float(y)
    except np.linalg.LinAlgError:
        print("WARNING: Singular homography matrix, using identity")
        return float(u), float(v)

def deduce_cam_and_frame(file_name: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Estrae video_id, frame_idx e cam_id dal nome file.
    
    Args:
        file_name: Nome del file immagine
        
    Returns:
        Tuple (video_id, frame_idx, cam_id)
    """
    base = Path(file_name).name
    
    # Pattern per video e frame
    m1 = re.search(r"([^_]+)_frame_(\d+)", base)
    m2 = re.search(r"([^_]+)_(\d+)", base) if not m1 else None
    
    video = m1.group(1) if m1 else (m2.group(1) if m2 else None)
    frame = int(m1.group(2)) if m1 else (int(m2.group(2)) if m2 else None)
    
    # Pattern per camera
    cam_match = re.search(r"cam[_-]?(\d+)", base, re.IGNORECASE)
    cam_id = f"cam_{cam_match.group(1)}" if cam_match else None
    
    return video, frame, cam_id

def load_gt(coco_json_path: str, video_name: str = VIDEO_NAME) -> Dict[Tuple[int, str], List[dict]]:
    """
    Carica i dati ground truth dal file COCO JSON.
    
    Args:
        coco_json_path: Percorso al file _annotations_rectified.coco.json
        video_name: Nome del video da processare
        
    Returns:
        Dizionario con chiavi (frame_idx, cam_id) e annotazioni GT
    """
    print(f"Loading ground truth from {coco_json_path} for video {video_name}")
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Mappa image_id -> info immagine
    images_by_id = {img['id']: img for img in coco_data['images']}
    
    # Raggruppa annotazioni per (frame_idx, cam_id)
    gt_by_frame_cam = defaultdict(list)
    
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        
        if img_id not in images_by_id:
            continue
            
        img_info = images_by_id[img_id]
        video, frame_idx, cam_id = deduce_cam_and_frame(img_info['file_name'])
        
        # Filtra per video specificato
        if video_name and video != video_name:
            continue
            
        if frame_idx is None or cam_id is None:
            continue
        
        # Converti classe
        original_class = ann['category_id']
        converted_class = gt_class_conversion(original_class)
        
        # Converti bbox da [x,y,w,h]
        bbox = ann['bbox']
        x, y, w, h = bbox
        
        # Usa bottom-center per player/referee, center per ball
        if converted_class in [1, 2]:  # player/referee
            u = x + w * 0.5
            v = y + h  # bottom
        else:  # ball
            u = x + w * 0.5
            v = y + h * 0.5  # center
        
        gt_annotation = {
            'u': u,
            'v': v,
            'class_id': converted_class,
            'original_class': original_class,
            'bbox': [x, y, x + w, y + h]
        }
        
        gt_by_frame_cam[(frame_idx, cam_id)].append(gt_annotation)
    
    print(f"Loaded GT for {len(gt_by_frame_cam)} (frame, camera) combinations")
    return dict(gt_by_frame_cam)

def align_frames(gt_by_frame_cam: Dict[Tuple[int, str], List[dict]], 
                predictions_df: pd.DataFrame) -> Dict[Tuple[int, str], List[dict]]:
    """
    Allinea temporalmente GT e predizioni usando FRAME_SCALE e FRAME_OFFSET.
    
    Args:
        gt_by_frame_cam: Dizionario con GT per (frame, cam)
        predictions_df: DataFrame con predizioni tracker
        
    Returns:
        GT allineato con chiavi (tracker_frame, cam_id)
    """
    print("Aligning frames between GT and predictions...")
    
    aligned_gt = {}
    tracker_frames_available = set(predictions_df['t'].unique())
    
    for (gt_frame_idx, cam_id), annotations in gt_by_frame_cam.items():
        # Calcola frame corrispondente nel tracker
        tracker_frame = FRAME_OFFSET + gt_frame_idx * FRAME_SCALE
        
        if tracker_frame in tracker_frames_available:
            aligned_gt[(tracker_frame, cam_id)] = annotations
    
    print(f"Aligned frames: {len(aligned_gt)} (frame, camera) combinations")
    if aligned_gt:
        frames = [t for t, _ in aligned_gt.keys()]
        print(f"Frame range: {min(frames)} - {max(frames)}")
    
    return aligned_gt

def load_homography_matrix(camera_data_path: str) -> Optional[np.ndarray]:
    """
    DEPRECATED: Usa load_calibrations_and_H invece.
    Mantienuta per compatibilità.
    """
    print("WARNING: load_homography_matrix is deprecated, use load_calibrations_and_H")
    return np.eye(3)

def project_gt_to_field(gt_by_frame_cam: Dict[Tuple[int, str], List[dict]], 
                       calibrations: Dict[str, dict]) -> Dict[int, List[dict]]:
    """
    Proietta i bounding box GT da pixel a coordinate campo usando omografie specifiche per camera.
    
    Args:
        gt_by_frame_cam: GT con coordinate pixel per (frame, cam)
        calibrations: Calibrazioni camere con matrici H
        
    Returns:
        GT aggregato per frame con coordinate campo
    """
    print("Projecting GT from pixel to field coordinates using per-camera homographies...")
    
    projected_gt_by_frame = defaultdict(list)
    
    for (frame_idx, cam_id), annotations in gt_by_frame_cam.items():
        if cam_id not in calibrations:
            print(f"WARNING: No calibration found for {cam_id}, skipping")
            continue
            
        H = calibrations[cam_id]["H"]
        
        for ann in annotations:
            u, v = ann['u'], ann['v']
            
            # Proietta su campo
            field_x, field_y = img_to_field_xy(H, u, v)
            
            # Crea annotazione proiettata
            projected_ann = ann.copy()
            projected_ann['field_x'] = field_x
            projected_ann['field_y'] = field_y
            projected_ann['cam_id'] = cam_id
            
            projected_gt_by_frame[frame_idx].append(projected_ann)
    
    print(f"Projected GT to {len(projected_gt_by_frame)} frames")
    return dict(projected_gt_by_frame)

def match_predictions_to_gt_hungarian(predictions: List[Dict], gt_annotations: List[Dict], 
                                    distance_threshold: float = 1.0) -> Tuple[List, List, List]:
    """
    Associa predizioni a GT usando algoritmo Hungarian con soglia di distanza.
    
    Args:
        predictions: Lista predizioni per un frame
        gt_annotations: Lista annotazioni GT per un frame
        distance_threshold: Soglia massima distanza in metri
        
    Returns:
        Tuple (matches, unmatched_pred, unmatched_gt)
    """
    if not predictions or not gt_annotations:
        return [], list(range(len(predictions))), list(range(len(gt_annotations)))
    
    # Matrice costi (distanze)
    cost_matrix = np.full((len(gt_annotations), len(predictions)), 1e6, dtype=float)
    
    for gt_idx, gt_ann in enumerate(gt_annotations):
        for pred_idx, pred in enumerate(predictions):
            # Distanza euclidea nel piano XY
            dx = pred['x'] - gt_ann['field_x']
            dy = pred['y'] - gt_ann['field_y']
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance <= distance_threshold:
                cost_matrix[gt_idx, pred_idx] = distance
    
    # Risolvi assegnamento con Hungarian
    try:
        from scipy.optimize import linear_sum_assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
    except ImportError:
        print("WARNING: scipy not available, falling back to greedy matching")
        return match_predictions_to_gt(predictions, gt_annotations, distance_threshold)
    
    # Estrai match validi
    matches = []
    for gt_idx, pred_idx in zip(row_indices, col_indices):
        if cost_matrix[gt_idx, pred_idx] < 1e6:
            matches.append((int(gt_idx), int(pred_idx), cost_matrix[gt_idx, pred_idx]))
    
    # Determina non-matched
    matched_gt = set(gt_idx for gt_idx, _, _ in matches)
    matched_pred = set(pred_idx for _, pred_idx, _ in matches)
    
    unmatched_gt = [i for i in range(len(gt_annotations)) if i not in matched_gt]
    unmatched_pred = [i for i in range(len(predictions)) if i not in matched_pred]
    
    return matches, unmatched_pred, unmatched_gt

def match_predictions_to_gt(predictions: List[Dict], gt_annotations: List[Dict], 
                           distance_threshold: float = 1.0) -> Tuple[List, List, List]:
    """
    Associa predizioni a GT usando distanza euclidea nel piano XY (versione greedy).
    
    Args:
        predictions: Lista predizioni per un frame
        gt_annotations: Lista annotazioni GT per un frame
        distance_threshold: Soglia massima distanza in metri
        
    Returns:
        Tuple (matches, unmatched_pred, unmatched_gt)
    """
    if not predictions or not gt_annotations:
        return [], list(range(len(predictions))), list(range(len(gt_annotations)))
    
    # Calcola matrice distanze
    distances = np.zeros((len(gt_annotations), len(predictions)))
    
    for gt_idx, gt_ann in enumerate(gt_annotations):
        for pred_idx, pred in enumerate(predictions):
            # Distanza euclidea nel piano XY
            dx = pred['x'] - gt_ann['field_x']
            dy = pred['y'] - gt_ann['field_y']
            distances[gt_idx, pred_idx] = np.sqrt(dx*dx + dy*dy)
    
    # Matching greedy basato su distanza minima
    matches = []
    unmatched_gt = list(range(len(gt_annotations)))
    unmatched_pred = list(range(len(predictions)))
    
    while unmatched_gt and unmatched_pred:
        # Trova coppia con distanza minima
        sub_distances = distances[np.ix_(unmatched_gt, unmatched_pred)]
        
        if sub_distances.size == 0 or np.min(sub_distances) > distance_threshold:
            break
            
        min_idx = np.unravel_index(np.argmin(sub_distances), sub_distances.shape)
        gt_idx = unmatched_gt[min_idx[0]]
        pred_idx = unmatched_pred[min_idx[1]]
        
        matches.append((gt_idx, pred_idx, distances[gt_idx, pred_idx]))
        unmatched_gt.remove(gt_idx)
        unmatched_pred.remove(pred_idx)
    
    return matches, unmatched_pred, unmatched_gt

def calculate_metrics(matches: List, distance_threshold: float) -> Dict:
    """
    Calcola metriche di posizione per i match trovati.
    
    Args:
        matches: Lista di (gt_idx, pred_idx, distance)
        distance_threshold: Soglia per TP/FP
        
    Returns:
        Dizionario con metriche
    """
    if not matches:
        return {
            'mae': 0.0,
            'rmse': 0.0,
            'percentiles': {'25': 0.0, '50': 0.0, '75': 0.0, '90': 0.0},
            'tp': 0,
            'errors': []
        }
    
    distances = [match[2] for match in matches]
    tp_count = sum(1 for d in distances if d <= distance_threshold)
    
    return {
        'mae': np.mean(distances),
        'rmse': np.sqrt(np.mean(np.square(distances))),
        'percentiles': {
            '25': np.percentile(distances, 25),
            '50': np.percentile(distances, 50),
            '75': np.percentile(distances, 75),
            '90': np.percentile(distances, 90)
        },
        'tp': tp_count,
        'errors': distances
    }

def evaluate_metrics(gt_by_frame: Dict, predictions_df: pd.DataFrame, 
                    distance_thresholds: List[float] = [0.5, 1.0, 1.5, 2.0]) -> Dict:
    """
    Valuta le metriche di tracking 3D per tutte le classi e soglie.
    
    Args:
        gt_by_frame: GT proiettato sul campo aggregato per frame
        predictions_df: DataFrame predizioni allineate
        distance_thresholds: Soglie distanza per evaluation
        
    Returns:
        Dizionario completo con tutte le metriche
    """
    print("Evaluating 3D tracking metrics...")
    
    class_names = {0: 'ball', 1: 'player', 2: 'referee'}
    results = {
        'overall': {},
        'by_class': {},
        'by_threshold': {}
    }
    
    # Inizializza contatori globali
    global_stats = {
        'total_matches': [],
        'total_tp': {thresh: 0 for thresh in distance_thresholds},
        'total_fp': {thresh: 0 for thresh in distance_thresholds},
        'total_fn': {thresh: 0 for thresh in distance_thresholds},
        'total_gt': 0,
        'total_pred': 0
    }
    
    # Statistiche per classe
    class_stats = {class_id: {
        'matches': [],
        'tp': {thresh: 0 for thresh in distance_thresholds},
        'fp': {thresh: 0 for thresh in distance_thresholds},
        'fn': {thresh: 0 for thresh in distance_thresholds},
        'total_gt': 0,
        'total_pred': 0
    } for class_id in class_names.keys()}
    
    # Processa ogni frame
    common_frames = sorted(set(gt_by_frame.keys()) & set(predictions_df['t'].unique()))
    
    for frame_idx in common_frames:
        gt_annotations = gt_by_frame[frame_idx]
        frame_predictions = predictions_df[predictions_df['t'] == frame_idx]
        
        # Raggruppa per classe
        for class_id in class_names.keys():
            class_name = class_names[class_id]
            
            # Filtra GT e predizioni per questa classe
            gt_class = [ann for ann in gt_annotations if ann['class_id'] == class_id]
            pred_class = []
            
            for _, pred_row in frame_predictions.iterrows():
                pred_class_name = pred_row['class']
                if (pred_class_name == class_name) or \
                   (class_id == 1 and pred_class_name == 'player') or \
                   (class_id == 0 and pred_class_name == 'ball') or \
                   (class_id == 2 and pred_class_name == 'referee'):
                    pred_class.append({
                        'x': pred_row['x'],
                        'y': pred_row['y'],
                        'z': pred_row['z'],
                        'track_id': pred_row['track_id']
                    })
            
            # Aggiorna contatori
            class_stats[class_id]['total_gt'] += len(gt_class)
            class_stats[class_id]['total_pred'] += len(pred_class)
            global_stats['total_gt'] += len(gt_class)
            global_stats['total_pred'] += len(pred_class)
            
            # Matching per ogni soglia
            for thresh in distance_thresholds:
                # Usa Hungarian se disponibile, altrimenti greedy
                try:
                    matches, unmatched_pred, unmatched_gt = match_predictions_to_gt_hungarian(
                        pred_class, gt_class, thresh)
                except:
                    matches, unmatched_pred, unmatched_gt = match_predictions_to_gt(
                        pred_class, gt_class, thresh)
                
                tp = len(matches)
                fp = len(unmatched_pred)
                fn = len(unmatched_gt)
                
                class_stats[class_id]['tp'][thresh] += tp
                class_stats[class_id]['fp'][thresh] += fp
                class_stats[class_id]['fn'][thresh] += fn
                
                global_stats['total_tp'][thresh] += tp
                global_stats['total_fp'][thresh] += fp
                global_stats['total_fn'][thresh] += fn
                
                # Salva distanze per metriche di posizione (solo soglia 1.0m)
                if thresh == 1.0:
                    for match in matches:
                        class_stats[class_id]['matches'].append(match)
                        global_stats['total_matches'].append(match)
    
    # Calcola metriche finali
    
    # Metriche globali
    global_metrics = calculate_metrics(global_stats['total_matches'], 1.0)
    results['overall']['position_metrics'] = global_metrics
    
    results['overall']['detection_metrics'] = {}
    for thresh in distance_thresholds:
        tp = global_stats['total_tp'][thresh]
        fp = global_stats['total_fp'][thresh]
        fn = global_stats['total_fn'][thresh]
        
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-6, precision + recall)
        
        results['overall']['detection_metrics'][f'threshold_{thresh}m'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    # Metriche per classe
    for class_id, class_name in class_names.items():
        class_metrics = calculate_metrics(class_stats[class_id]['matches'], 1.0)
        
        results['by_class'][class_name] = {
            'position_metrics': class_metrics,
            'detection_metrics': {}
        }
        
        for thresh in distance_thresholds:
            tp = class_stats[class_id]['tp'][thresh]
            fp = class_stats[class_id]['fp'][thresh]
            fn = class_stats[class_id]['fn'][thresh]
            
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-6, precision + recall)
            
            results['by_class'][class_name]['detection_metrics'][f'threshold_{thresh}m'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
    
    print(f"Evaluation completed on {len(common_frames)} frames")
    return results

def print_summary(results: Dict):
    """
    Stampa un riassunto leggibile delle metriche.
    
    Args:
        results: Dizionario con tutti i risultati
    """
    print("\n" + "="*80)
    print("3D TRACKING EVALUATION SUMMARY")
    print("="*80)
    
    # Metriche globali
    print("\nOVERALL PERFORMANCE:")
    print("-" * 40)
    
    pos_metrics = results['overall']['position_metrics']
    print(f"Position Accuracy (1.0m threshold):")
    print(f"  MAE:  {pos_metrics['mae']:.3f} meters")
    print(f"  RMSE: {pos_metrics['rmse']:.3f} meters")
    print(f"  Median Error: {pos_metrics['percentiles']['50']:.3f} meters")
    print(f"  90th Percentile: {pos_metrics['percentiles']['90']:.3f} meters")
    
    print(f"\nDetection Performance:")
    for thresh_name, metrics in results['overall']['detection_metrics'].items():
        thresh = thresh_name.split('_')[1]
        print(f"  At {thresh}:")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall:    {metrics['recall']:.3f}")
        print(f"    F1:        {metrics['f1']:.3f}")
    
    # Metriche per classe
    print(f"\nPER-CLASS PERFORMANCE:")
    print("-" * 40)
    
    for class_name, class_results in results['by_class'].items():
        print(f"\n{class_name.upper()}:")
        
        pos_metrics = class_results['position_metrics']
        print(f"  Position MAE: {pos_metrics['mae']:.3f}m")
        print(f"  Position RMSE: {pos_metrics['rmse']:.3f}m")
        
        # Mostra solo F1 a 1.0m per brevità
        det_metrics = class_results['detection_metrics'].get('threshold_1.0m', {})
        if det_metrics:
            print(f"  F1 Score (1.0m): {det_metrics['f1']:.3f}")
            print(f"  TP: {det_metrics['tp']}, FP: {det_metrics['fp']}, FN: {det_metrics['fn']}")

def save_results(results: Dict, output_path: str):
    """
    Salva i risultati in formato JSON.
    
    Args:
        results: Dizionario con tutti i risultati
        output_path: Percorso file di output
    """
    print(f"\nSaving results to {output_path}")
    
    # Converti numpy arrays in liste per serializzazione JSON
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    serializable_results = convert_numpy(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

def main():
    """Funzione principale che esegue l'intera pipeline di evaluation."""
    
    # Percorsi file (configura secondo la tua struttura)
    base_path = r"c:\Users\nicol\Desktop\CV_Tracking"
    
    tracks_csv = os.path.join(base_path, "3D_tracking", "tracks3d", "tracks3d.csv")
    gt_json = os.path.join(base_path, "GroundTruthData", "train", "_annotations_rectified.coco.json")
    camera_data = os.path.join(base_path, "support_material", "3D_tracking_material", "camera_data")
    output_json = os.path.join(base_path, "3D_tracking", "metrics_summary.json")
    
    print("Starting 3D Tracking Evaluation Pipeline")
    print(f"Video: {VIDEO_NAME}")
    print(f"Tracks CSV: {tracks_csv}")
    print(f"GT JSON: {gt_json}")
    
    # 1. Carica predizioni
    predictions_df = load_predictions(tracks_csv)
    if predictions_df.empty:
        print("Failed to load predictions. Exiting.")
        return
    
    # 2. Carica calibrazioni camere
    try:
        calibrations = load_calibrations_and_H(camera_data)
        print(f"Loaded calibrations for cameras: {list(calibrations.keys())}")
    except Exception as e:
        print(f"Error loading camera calibrations: {e}")
        print("Using identity transformation as fallback")
        calibrations = {"cam_default": {"H": np.eye(3)}}
    
    # 3. Carica ground truth
    gt_by_frame_cam = load_gt(gt_json, VIDEO_NAME)
    if not gt_by_frame_cam:
        print("Failed to load ground truth. Exiting.")
        return
    
    # 4. Allinea frame
    gt_aligned = align_frames(gt_by_frame_cam, predictions_df)
    if not gt_aligned:
        print("No aligned frames found. Check frame alignment parameters.")
        return
    
    # 5. Proietta GT su campo usando omografie per-camera
    gt_projected = project_gt_to_field(gt_aligned, calibrations)
    if not gt_projected:
        print("Failed to project GT to field coordinates. Exiting.")
        return
    
    # 6. Filtra predizioni per frame allineati
    aligned_frames = list(gt_projected.keys())
    predictions_aligned = predictions_df[predictions_df['t'].isin(aligned_frames)].copy()
    
    print(f"Final aligned data: {len(aligned_frames)} frames, {len(predictions_aligned)} predictions")
    
    # 7. Calcola metriche
    results = evaluate_metrics(gt_projected, predictions_aligned)
    
    # 8. Mostra e salva risultati
    print_summary(results)
    save_results(results, output_json)
    
    print(f"\nEvaluation completed successfully!")
    print(f"Results saved to: {output_json}")

if __name__ == "__main__":
    main()