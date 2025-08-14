import os
import json
import re
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

video = "out4"

def gt_class_conversion(class_id):
    """
    Convert ground truth class IDs to a simpler set of classes:
    - Class 1 -> Class 0
    - Class 6,7 -> Class 2
    - All others -> Class 1
    
    Args:
        class_id (int): Original class ID (1-13)
        
    Returns:
        int: Converted class ID (0-2)
    """
    if class_id == 1:
        return 0
    elif class_id in [7, 8]:
        return 2
    else:
        return 1

def load_coco_annotations(coco_json_path):
    """
    Load and parse COCO format annotations, filtering only for the specified video
    and ensuring frames are ordered correctly for comparison with tracker results.
    
    Note: The ground truth frame indices are not sequential.
    The mapping is as follows:
    - GT frame 1 corresponds to tracker frame 3
    - GT frame 2 corresponds to tracker frame 8
    - GT frame 3 corresponds to tracker frame 13
    and so on (with a pattern of +5 between consecutive frames)
    """
    # Get the global video variable
    global video
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create image_id to image info mapping
    images_by_id = {}
    for img in coco_data['images']:
        images_by_id[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }
    
    # Extract video and frame information from filenames
    # Example pattern: out13_frame_0001_png.rf.eee79b00ea46b4cec14642e05f7d5ac9.jpg
    frame_pattern = re.compile(r'([^_]+)_frame_(\d+)_')
    
    # Alternative pattern if the above doesn't match
    alt_pattern = re.compile(r'([^_]+)_(\d+)')
    
    # Map image_id to (video_id, frame_idx)
    video_frame_map = {}
    for img_id, img_info in images_by_id.items():
        file_name = img_info['file_name']
        
        # Try the primary pattern first
        match = frame_pattern.search(file_name)
        if match:
            video_id = match.group(1)  # e.g. "out13"
            frame_idx = int(match.group(2))  # e.g. 1
        else:
            # Try alternative pattern
            match = alt_pattern.search(file_name)
            if match:
                video_id = match.group(1)
                frame_idx = int(match.group(2))
            else:
                # If no pattern matches, use the whole filename as video_id and set frame_idx to 0
                video_id = os.path.splitext(file_name)[0]
                frame_idx = 0
        
        # Only store frames from the specified video
        if video_id == video:
            # Calculate the actual frame index in the tracker's frame sequence
            # GT frame n corresponds to tracker frame -2 + n*5
            # For example:
            # GT frame 1 -> tracker frame 3 = -2 + 1*5
            # GT frame 2 -> tracker frame 8 = -2 + 2*5
            actual_frame_idx = -2 + (frame_idx * 5)
            video_frame_map[img_id] = (video_id, actual_frame_idx)
    
    # Group annotations by (video_id, frame_idx), only for the target video
    gt_by_frame = defaultdict(list)
    
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        
        # Only process annotations for our target video
        if img_id in video_frame_map:
            video_id, frame_idx = video_frame_map[img_id]
            
            # Convert COCO bbox [x,y,w,h] to [x1,y1,x2,y2]
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x, y, x+w, y+h
            
            gt_by_frame[(video_id, frame_idx)].append({
                'bbox_xyxy': [x1, y1, x2, y2],
                'class_id': ann['category_id'],
                'area': ann['area'],
                'id': ann['id']
            })
    
    # Create a mapping of category_id to category_name
    categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
    
    print(f"Loaded {len(gt_by_frame)} frames for video '{video}'")
    
    # Sort the frames by their index to ensure proper temporal order
    sorted_frames = sorted(gt_by_frame.items(), key=lambda x: x[0][1])
    ordered_gt_by_frame = defaultdict(list)
    for key, value in sorted_frames:
        ordered_gt_by_frame[key] = value
    
    return ordered_gt_by_frame, images_by_id, categories



def load_tracker_results(tracker_results_path):
    """
    Load tracking results from JSON file
    Expected format per frame:
      { "frame": int, "tracks": [ {"confidence": float, "bbox": [x1,y1,x2,y2], "class_id": int, "id": int}, ... ] }
    """
    tracker_results = defaultdict(list)
    with open(tracker_results_path, 'r') as f:
        json_data = json.load(f)

    video_id = os.path.basename(tracker_results_path).split('_')[0]

    for frame_data in json_data:
        if "frame" not in frame_data:
            continue
        frame_idx = frame_data.get("frame")
        tracks = frame_data.get("tracks", [])
        if not tracks:
            continue

        for track in tracks:
            bbox_xyxy = track.get("bbox")
            if not bbox_xyxy or len(bbox_xyxy) != 4:
                continue  # salta track senza bbox valida

            x1, y1, x2, y2 = bbox_xyxy
            class_id = int(track.get("class_id", 0))
            confidence = float(track.get("confidence", 0.0))
            track_id = track.get("id")  # fix: usa il campo 'id' del JSON
            if track_id is None:
                continue  # richiedi un id persistente per tracking

            tracker_results[(video_id, frame_idx)].append({
                'bbox_xyxy': [x1, y1, x2, y2],
                'class_id': class_id,
                'obj_id': int(track_id),
                'confidence': confidence
            })

    return tracker_results

def calculate_iou(box1, box2):
    """
    IoU tra due box [x1, y1, x2, y2]. Ritorna 0 per box non valide.
    """
    # box non valide (x2<=x1 o y2<=y1) -> IoU 0
    if box1[2] <= box1[0] or box1[3] <= box1[1] or box2[2] <= box2[0] or box2[3] <= box2[1]:
        return 0.0

    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    width_inter = max(0.0, x2_inter - x1_inter)
    height_inter = max(0.0, y2_inter - y1_inter)
    area_inter = width_inter * height_inter

    area_box1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area_box2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union = area_box1 + area_box2 - area_inter
    if union <= 0:
        return 0.0
    return area_inter / union

def quick_iou_self_test():
    """Asserzioni su casi noti per validare IoU."""
    def almost(a,b,eps=1e-6): return abs(a-b) < eps
    assert almost(calculate_iou([0,0,10,10],[0,0,10,10]), 1.0)
    assert almost(calculate_iou([0,0,10,10],[20,20,30,30]), 0.0)
    # overlap 5x5 -> 25, union 100+100-25=175 => 25/175=0.142857...
    assert almost(calculate_iou([0,0,10,10],[5,5,15,15]), 25/175)
    # contenuta: 6x6=36, union 100 -> 0.36
    assert almost(calculate_iou([0,0,10,10],[2,2,8,8]), 0.36)
    # box non valida -> 0
    assert almost(calculate_iou([10,10,5,5],[0,0,10,10]), 0.0)
    print("IoU self-test: OK")
def match_detections(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Match ground truth boxes with predicted boxes based on IoU
    """
    matches = []  # List of (gt_idx, pred_idx)
    unmatched_gt = list(range(len(gt_boxes)))
    unmatched_pred = list(range(len(pred_boxes)))
    
    # Calculate IoU matrix
    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for gt_idx in range(len(gt_boxes)):
        for pred_idx in range(len(pred_boxes)):
            # Only consider matching if class IDs are the same
            if gt_class_conversion(gt_boxes[gt_idx]['class_id']) == pred_boxes[pred_idx]['class_id']:
                iou_matrix[gt_idx, pred_idx] = calculate_iou(
                    gt_boxes[gt_idx]['bbox_xyxy'], 
                    pred_boxes[pred_idx]['bbox_xyxy']
                )
    
    # Match detections greedily
    while True:
        # Find highest IoU
        if not unmatched_gt or not unmatched_pred:
            break
            
        # Get indices of highest remaining IoU
        sub_iou = iou_matrix[unmatched_gt][:, unmatched_pred]
        if sub_iou.size == 0 or np.max(sub_iou) < iou_threshold:
            break
            
        i, j = np.unravel_index(np.argmax(sub_iou), sub_iou.shape)
        gt_idx = unmatched_gt[i]
        pred_idx = unmatched_pred[j]
        
        # Add match
        matches.append((gt_idx, pred_idx))
        
        # Remove matched indices
        unmatched_gt.remove(gt_idx)
        unmatched_pred.remove(pred_idx)
    
    return matches, unmatched_gt, unmatched_pred

def evaluate_tracking(gt_by_frame, tracker_results, iou_threshold=0.5, link_iou=0.5):
    """
    Evaluate tracking performance using standard metrics
    """
    # crea identità GT consistenti
    gt_by_frame = build_gt_tracks(gt_by_frame, link_iou=link_iou)

    total_gt = total_pred = total_matched = 0
    total_fp = total_fn = total_id_switches = total_fragments = 0
    total_iou = 0.0

    gt_track_history = {}
    matched_by_frame = defaultdict(set)  # (vid, fidx) -> set(gt_tid)  <-- NEW

    # solo frame comuni
    all_frames = sorted(set(gt_by_frame.keys()).intersection(set(tracker_results.keys())))
    if not all_frames:
        print("Warning: No common frames found between ground truth and tracker results.")

    for (vid, fidx) in all_frames:
        gt_boxes = gt_by_frame.get((vid, fidx), [])
        pred_boxes = tracker_results.get((vid, fidx), [])

        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        matches, unmatched_gt, unmatched_pred = match_detections(gt_boxes, pred_boxes, iou_threshold)

        total_matched += len(matches)
        total_fp += len(unmatched_pred)
        total_fn += len(unmatched_gt)

        for gt_idx, pred_idx in matches:
            g = gt_boxes[gt_idx]
            p = pred_boxes[pred_idx]

            total_iou += calculate_iou(g['bbox_xyxy'], p['bbox_xyxy'])

            gt_tid = g.get('tid')
            pred_id = p.get('obj_id')

            # registra match per fragments  <-- NEW
            if gt_tid is not None:
                matched_by_frame[(vid, fidx)].add(gt_tid)

            # ID switches
            if gt_tid is not None and pred_id is not None:
                last = gt_track_history.get(gt_tid)
                if last is not None and pred_id != last['pred_id']:
                    total_id_switches += 1
                gt_track_history[gt_tid] = {'pred_id': pred_id, 'frame_idx': fidx}

    # === Fragments: numero di segmenti matched per GT - 1 ===  <-- NEW
    # presenza per ogni gt_tid (limitata ai frame comuni per coerenza con le altre metriche)
    presence_by_tid = defaultdict(list)  # tid -> [(vid,fidx), ...] in ordine
    for key in all_frames:
        for g in gt_by_frame.get(key, []):
            if 'tid' in g:
                presence_by_tid[g['tid']].append(key)

    for tid, frame_keys in presence_by_tid.items():
        segments = 0
        inside = False
        for fk in frame_keys:
            is_matched = tid in matched_by_frame.get(fk, set())
            if is_matched and not inside:
                segments += 1
                inside = True
            elif not is_matched and inside:
                inside = False
        if segments > 0:
            total_fragments += max(0, segments - 1)

    mota = 1 - (total_fp + total_fn + total_id_switches) / max(1, total_gt)
    motp = total_iou / max(1, total_matched)
    precision = total_matched / max(1, total_matched + total_fp)
    recall = total_matched / max(1, total_gt)
    f1_score = 2 * precision * recall / max(1e-6, precision + recall)

    return {
        'MOTA': mota,
        'MOTP': motp,
        'Precision': precision,
        'Recall': recall,
        'F1': f1_score,
        'ID_Switches': total_id_switches,
        'Fragments': total_fragments,
        'FP': total_fp,
        'FN': total_fn,
        'GT_total': total_gt,
        'Matches': total_matched
    }

def visualize_metrics(metrics, title="Tracking Performance"):
    """
    Visualize tracking metrics
    """
    # Plot bar chart for main metrics
    main_metrics = ['MOTA', 'MOTP', 'Precision', 'Recall', 'F1']
    values = [metrics[m] for m in main_metrics]
    
    plt.figure(figsize=(10, 6))
    plt.bar(main_metrics, values)
    plt.title(title)
    plt.ylabel('Score')
    plt.ylim([0, 1])
    
    # Add values on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    #plt.savefig('tracking_metrics.png')
    plt.show()
    
    # Plot error analysis
    error_metrics = ['ID_Switches', 'Fragments', 'FP', 'FN']
    error_values = [metrics[m] for m in error_metrics]
    
    plt.figure(figsize=(10, 6))
    plt.bar(error_metrics, error_values)
    plt.title(f"{title} - Error Analysis")
    plt.ylabel('Count')
    
    # Add values on top of bars
    for i, v in enumerate(error_values):
        plt.text(i, v + 2, str(v), ha='center')
    
    plt.tight_layout()
    #plt.savefig('tracking_errors.png')
    plt.show()


def debug_ious(gt_by_frame, tracker_results, max_samples=10, iou_threshold=0.5):
    """Stampa IoU e box per i primi match trovati su frame comuni."""
    printed = 0
    common = sorted(set(gt_by_frame.keys()) & set(tracker_results.keys()))
    for fk in common:
        gt = gt_by_frame.get(fk, [])
        pr = tracker_results.get(fk, [])
        matches, _, _ = match_detections(gt, pr, iou_threshold)
        for gi, pi in matches:
            g = gt[gi]; p = pr[pi]
            iou = calculate_iou(g['bbox_xyxy'], p['bbox_xyxy'])
            print(f"Frame {fk}: IoU={iou:.4f} | GT {g['bbox_xyxy']} (cls {gt_class_conversion(g['class_id'])}) "
                  f"vs PR {p['bbox_xyxy']} (cls {p['class_id']})")
            printed += 1
            if printed >= max_samples:
                return

def build_gt_tracks(gt_by_frame, link_iou=0.5):
    """
    Assegna un 'tid' (track id GT) consistente alle box GT collegando tra frame consecutivi
    oggetti della stessa classe (dopo conversione) con IoU >= link_iou.
    Modifica in-place i dizionari delle box aggiungendo 'tid'.
    """
    # raggruppa per video e ordina per frame_idx
    frames_per_video = defaultdict(list)
    for (vid, fidx) in gt_by_frame.keys():
        frames_per_video[vid].append(fidx)
    for vid in frames_per_video:
        frames_per_video[vid] = sorted(set(frames_per_video[vid]))

    next_tid = 0
    prev_boxes_by_vid = {}  # vid -> list of dicts from prev frame (con 'tid')

    for vid, frame_list in frames_per_video.items():
        prev_boxes = []
        for fidx in frame_list:
            key = (vid, fidx)
            curr = gt_by_frame.get(key, [])
            if not prev_boxes:
                # assegna nuovi tid a tutti
                for b in curr:
                    b['tid'] = next_tid
                    next_tid += 1
                prev_boxes = curr
                prev_boxes_by_vid[vid] = prev_boxes
                continue

            # costruiamo matrice IoU solo tra stessa classe convertita
            M = np.zeros((len(prev_boxes), len(curr)), dtype=float)
            for i, pb in enumerate(prev_boxes):
                for j, cb in enumerate(curr):
                    if gt_class_conversion(pb['class_id']) != gt_class_conversion(cb['class_id']):
                        M[i, j] = 0.0
                    else:
                        M[i, j] = calculate_iou(pb['bbox_xyxy'], cb['bbox_xyxy'])

            # matching greedy maggiore IoU
            used_prev = set()
            used_curr = set()
            while True:
                if M.size == 0:
                    break
                i, j = np.unravel_index(np.argmax(M), M.shape)
                if M[i, j] < link_iou:
                    break
                if i in used_prev or j in used_curr:
                    M[i, j] = -1.0
                    continue
                # assegna tid del prev al current
                curr[j]['tid'] = prev_boxes[i]['tid']
                used_prev.add(i)
                used_curr.add(j)
                M[i, j] = -1.0

            # i non assegnati ricevono nuovi tid
            for j, cb in enumerate(curr):
                if j not in used_curr:
                    cb['tid'] = next_tid
                    next_tid += 1

            prev_boxes = curr
            prev_boxes_by_vid[vid] = prev_boxes

    return gt_by_frame

if __name__ == "__main__":
    # opzionale: autovalidazione IoU
    quick_iou_self_test()
    # Load ground truth data
    coco_json_path = os.path.join("GroundTruthData", "train", "_annotations.coco.json")
    gt_by_frame, images_by_id, categories = load_coco_annotations(coco_json_path)
    
    # Example: Load tracking results (adjust path as needed)
    tracker_results_path = os.path.join("2D_tracking", "tracked", f"{video}_tracks.json")

    # Check if tracker results exist
    if os.path.exists(tracker_results_path):
        tracker_results = load_tracker_results(tracker_results_path)
        # stampa 10 IoU di esempio
        #debug_ious(gt_by_frame, tracker_results, max_samples=10, iou_threshold=0.5)
        
        # Evaluate tracking performance
        metrics = evaluate_tracking(gt_by_frame, tracker_results)
        
        # Print metrics
        print("\nTracking Performance Metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}" if isinstance(value, float) else f"{metric_name}: {value}")
        
        # Visualize metrics
        #visualize_metrics(metrics)
    else:
        print(f"\nTracker results file not found: {tracker_results_path}")
        print("Please provide tracking results in the expected format to evaluate performance")
        print("\nExample format for tracker results (CSV):")
        print("video_id,frame_id,obj_id,x1,y1,x2,y2,class_id,confidence")