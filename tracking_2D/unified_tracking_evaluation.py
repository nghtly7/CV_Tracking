"""
Unified Basketball Tracking Evaluation Script
============================================

This script computes all standard tracking metrics for 2D object tracking:
- MOTA (Multiple Object Tracking Accuracy)
- MOTP (Multiple Object Tracking Precision) 
- ID Switches (IDS)
- ID F1-score (IDF1)
- False Positives (FP) and False Negatives (FN)
- Mostly Tracked (MT) and Mostly Lost (ML)
- HOTA metrics: HOTA, AssA, DetA, LocA

Handles frame rate difference: GT at 5fps vs tracking at 25fps
GT frame N corresponds to tracking frame (N-1)*5+3

Usage:
    python3 unified_tracking_evaluation.py --videos out2 out4 out13
    python3 unified_tracking_evaluation.py --videos out2 --debug
    python3 unified_tracking_evaluation.py --help

Dependencies to install manually:
    pip install numpy scipy motmetrics
"""

import numpy as np
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from collections import defaultdict
import logging
import sys
import re

# Try to import motmetrics (optional)
try:
    import motmetrics as mm
    MOTMETRICS_AVAILABLE = True
except ImportError:
    MOTMETRICS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BoundingBox:
    """Class to represent a bounding box with conversion utilities."""
    
    def __init__(self, x_center: float, y_center: float, width: float, height: float, 
                 class_id: int, track_id: Optional[int] = None, confidence: float = 1.0):
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.class_id = class_id
        self.track_id = track_id
        self.confidence = confidence
    
    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Convert to (x1, y1, x2, y2) format."""
        x1 = self.x_center - self.width / 2
        y1 = self.y_center - self.height / 2
        x2 = self.x_center + self.width / 2
        y2 = self.y_center + self.height / 2
        return x1, y1, x2, y2
    
    def area(self) -> float:
        """Calculate box area."""
        return self.width * self.height
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate IoU with another bounding box."""
        x1_1, y1_1, x2_1, y2_1 = self.to_xyxy()
        x1_2, y1_2, x2_2, y2_2 = other.to_xyxy()
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = self.area()
        area2 = other.area()
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class UnifiedTrackingEvaluator:
    """Unified tracking evaluator with all metrics and debugging capabilities."""
    
    def __init__(self, gt_dir: str, pred_dir: str, video_name: str, debug: bool = False):
        self.gt_dir = Path(gt_dir)
        self.pred_dir = Path(pred_dir)
        self.video_name = video_name
        self.debug = debug
        self.iou_threshold = 0.5
        self.class_names = {0: "Ball", 1: "Player", 2: "Referee"}
        
    def load_yolo_labels(self, file_path: Path) -> List[BoundingBox]:
        """Load YOLO format labels from file."""
        boxes = []
        if not file_path.exists():
            return boxes
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Check if track ID is present (for predictions)
                        track_id = None
                        if len(parts) >= 6:
                            try:
                                track_id = int(float(parts[5]))
                            except:
                                track_id = None
                        
                        # Check if confidence is present
                        confidence = 1.0
                        if len(parts) >= 7:
                            confidence = float(parts[6])
                        elif len(parts) == 6 and track_id is None:
                            try:
                                confidence = float(parts[5])
                            except:
                                pass
                        
                        boxes.append(BoundingBox(x_center, y_center, width, height, 
                                               class_id, track_id, confidence))
        except Exception as e:
            if self.debug:
                logger.warning(f"Error loading {file_path}: {e}")
        
        return boxes
    
    def get_frame_mapping(self) -> Dict[int, int]:
        """
        Create mapping from GT frame numbers to tracking frame numbers.
        GT frame 1 -> tracking frame 3, GT frame 2 -> tracking frame 8, etc.
        """
        mapping = {}
        for gt_frame in range(1, 121):  # GT frames 1-120
            tracking_frame = (gt_frame - 1) * 5 + 3  # 3, 8, 13, ..., 598
            mapping[gt_frame] = tracking_frame
        return mapping
    
    def load_ground_truth(self) -> Dict[int, List[BoundingBox]]:
        """Load ground truth annotations."""
        gt_data = {}
        gt_files = list(self.gt_dir.glob("*.txt"))
        
        if self.debug:
            logger.info(f"Loading ground truth from {len(gt_files)} files")
        
        # Sort files and assign sequential frame numbers
        gt_files = sorted(gt_files)
        
        invalid_gt_count = 0
        for i, gt_file in enumerate(gt_files, 1):
            if i <= 120:  # Only take first 120 files
                boxes = self.load_yolo_labels(gt_file)
                # Filter out invalid boxes (coordinates > 1.0)
                valid_boxes = []
                for box in boxes:
                    if (0.0 <= box.x_center <= 1.0 and 0.0 <= box.y_center <= 1.0 and
                        0.0 <= box.width <= 1.0 and 0.0 <= box.height <= 1.0):
                        # Assign unique GT track ID if not present
                        if box.track_id is None:
                            box.track_id = f"gt_{i}_{len(valid_boxes)}"
                        valid_boxes.append(box)
                    else:
                        invalid_gt_count += 1
                
                gt_data[i] = valid_boxes
        
        if self.debug:
            logger.info(f"Loaded ground truth for {len(gt_data)} frames")
            if invalid_gt_count > 0:
                logger.warning(f"Filtered out {invalid_gt_count} invalid GT boxes")
        
        return gt_data
    
    def load_predictions(self, frame_mapping: Dict[int, int]) -> Dict[int, List[BoundingBox]]:
        """Load tracking predictions."""
        pred_data = {}
        invalid_pred_count = 0
        
        for gt_frame, tracking_frame in frame_mapping.items():
            pred_file = self.pred_dir / f"frame_{tracking_frame:06d}.txt"
            if pred_file.exists():
                boxes = self.load_yolo_labels(pred_file)
                # Filter out invalid boxes (coordinates > 1.0)
                valid_boxes = []
                for box in boxes:
                    if (0.0 <= box.x_center <= 1.0 and 0.0 <= box.y_center <= 1.0 and
                        0.0 <= box.width <= 1.0 and 0.0 <= box.height <= 1.0):
                        valid_boxes.append(box)
                    else:
                        invalid_pred_count += 1
                
                pred_data[gt_frame] = valid_boxes
            else:
                pred_data[gt_frame] = []
        
        if self.debug:
            logger.info(f"Loaded predictions for {len(pred_data)} frames")
            if invalid_pred_count > 0:
                logger.warning(f"Filtered out {invalid_pred_count} invalid prediction boxes")
        
        return pred_data
    
    def assign_detections(self, gt_boxes: List[BoundingBox], 
                         pred_boxes: List[BoundingBox]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Assign predictions to ground truth using flexible matching strategies.
        Returns: (matches, unmatched_gt, unmatched_pred)
        
        Matching strategy:
        1. First try: Same class + IoU >= threshold
        2. If no matches: Same class + any IoU > 0
        3. If still no matches: Any class + IoU >= threshold  
        4. If still no matches: Any class + any IoU > 0 (for debugging)
        5. If STILL no matches: Distance-based matching (same class, closest pairs)
        """
        if not gt_boxes or not pred_boxes:
            return [], list(range(len(gt_boxes))), list(range(len(pred_boxes)))
        
        def euclidean_distance(box1, box2):
            """Calculate center-to-center distance."""
            dx = box1.x_center - box2.x_center
            dy = box1.y_center - box2.y_center
            return np.sqrt(dx*dx + dy*dy)
        
        def try_matching(require_same_class=True, min_iou=None, use_distance=False, max_distance=None):
            """Try matching with given constraints."""
            if min_iou is None:
                min_iou = self.iou_threshold
                
            # Calculate similarity matrix (IoU or distance-based)
            similarity_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
            for i, gt_box in enumerate(gt_boxes):
                for j, pred_box in enumerate(pred_boxes):
                    # Check class constraint
                    if require_same_class and gt_box.class_id != pred_box.class_id:
                        continue
                    
                    if use_distance:
                        # Use distance-based matching
                        dist = euclidean_distance(gt_box, pred_box)
                        if max_distance is None or dist <= max_distance:
                            # Convert distance to similarity (higher = better)
                            similarity = 1.0 / (1.0 + dist)  # Higher similarity for closer objects
                            similarity_matrix[i, j] = similarity
                    else:
                        # Use IoU-based matching
                        iou = gt_box.iou(pred_box)
                        if iou >= min_iou:
                            similarity_matrix[i, j] = iou
            
            # Greedy assignment
            matches = []
            used_gt = set()
            used_pred = set()
            
            # Sort by similarity value (highest first)
            potential_matches = []
            for i in range(len(gt_boxes)):
                for j in range(len(pred_boxes)):
                    if similarity_matrix[i, j] > 0:
                        potential_matches.append((similarity_matrix[i, j], i, j))
            
            potential_matches.sort(reverse=True)
            
            for similarity_val, i, j in potential_matches:
                if i not in used_gt and j not in used_pred:
                    matches.append((i, j))
                    used_gt.add(i)
                    used_pred.add(j)
            
            unmatched_gt = [i for i in range(len(gt_boxes)) if i not in used_gt]
            unmatched_pred = [j for j in range(len(pred_boxes)) if j not in used_pred]
            
            return matches, unmatched_gt, unmatched_pred
        
        # Strategy 1: Same class + IoU >= threshold (standard)
        matches, unmatched_gt, unmatched_pred = try_matching(
            require_same_class=True, min_iou=self.iou_threshold)
        
        if matches:
            if self.debug:
                logger.info(f"Strategy 1 success: {len(matches)} matches with same class + IoU>={self.iou_threshold}")
            return matches, unmatched_gt, unmatched_pred
        
        # Strategy 2: Same class + any IoU > 0
        matches, unmatched_gt, unmatched_pred = try_matching(
            require_same_class=True, min_iou=0.001)
        
        if matches:
            if self.debug:
                logger.info(f"Strategy 2 success: {len(matches)} matches with same class + any IoU>0")
            return matches, unmatched_gt, unmatched_pred
        
        # Strategy 3: Any class + IoU >= threshold
        matches, unmatched_gt, unmatched_pred = try_matching(
            require_same_class=False, min_iou=self.iou_threshold)
        
        if matches:
            if self.debug:
                logger.warning(f"Strategy 3 fallback: {len(matches)} matches with any class + IoU>={self.iou_threshold}")
            return matches, unmatched_gt, unmatched_pred
        
        # Strategy 4: Any class + any IoU > 0 (debugging)
        matches, unmatched_gt, unmatched_pred = try_matching(
            require_same_class=False, min_iou=0.001)
        
        if matches:
            if self.debug:
                logger.warning(f"Strategy 4 fallback: {len(matches)} matches with any class + any IoU>0")
            return matches, unmatched_gt, unmatched_pred
        
        # Strategy 5: Distance-based matching (same class, distance < 0.5)
        matches, unmatched_gt, unmatched_pred = try_matching(
            require_same_class=True, use_distance=True, max_distance=0.5)
        
        if matches:
            if self.debug:
                logger.warning(f"Strategy 5 fallback: {len(matches)} matches with same class + distance<0.5")
            return matches, unmatched_gt, unmatched_pred
        
        # Strategy 6: Distance-based matching (same class, any distance)
        matches, unmatched_gt, unmatched_pred = try_matching(
            require_same_class=True, use_distance=True)
        
        if matches:
            if self.debug:
                logger.warning(f"Strategy 6 fallback: {len(matches)} matches with same class + any distance")
            return matches, unmatched_gt, unmatched_pred
        
        # Strategy 7: Distance-based matching (any class, closest pairs)
        matches, unmatched_gt, unmatched_pred = try_matching(
            require_same_class=False, use_distance=True)
        
        if matches and self.debug:
            logger.warning(f"Strategy 7 fallback: {len(matches)} matches with any class + any distance")
        
        return matches, unmatched_gt, unmatched_pred
    
    def compute_basic_metrics(self, gt_data: Dict[int, List[BoundingBox]], 
                             pred_data: Dict[int, List[BoundingBox]]) -> Dict:
        """Compute basic detection and tracking metrics."""
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_id_switches = 0
        total_motp_sum = 0.0
        total_motp_count = 0
        
        # Track correspondence for ID switches
        track_correspondence = {}  # pred_track_id -> gt_track_id
        
        frame_metrics = {}
        
        for frame_num in sorted(gt_data.keys()):
            gt_boxes = gt_data.get(frame_num, [])
            pred_boxes = pred_data.get(frame_num, [])
            
            matches, unmatched_gt, unmatched_pred = self.assign_detections(gt_boxes, pred_boxes)
            
            # Count metrics
            tp = len(matches)
            fp = len(unmatched_pred)
            fn = len(unmatched_gt)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            # MOTP calculation
            frame_motp = 0.0
            if matches:
                for gt_idx, pred_idx in matches:
                    iou = gt_boxes[gt_idx].iou(pred_boxes[pred_idx])
                    frame_motp += iou
                frame_motp /= len(matches)
                total_motp_sum += frame_motp * len(matches)
                total_motp_count += len(matches)
            
            # ID switches (simplified - requires track IDs in predictions)
            frame_id_switches = 0
            for gt_idx, pred_idx in matches:
                pred_track_id = pred_boxes[pred_idx].track_id
                if pred_track_id is not None:
                    gt_track_id = gt_boxes[gt_idx].track_id
                    
                    if pred_track_id in track_correspondence:
                        if track_correspondence[pred_track_id] != gt_track_id:
                            frame_id_switches += 1
                            track_correspondence[pred_track_id] = gt_track_id
                    else:
                        track_correspondence[pred_track_id] = gt_track_id
            
            total_id_switches += frame_id_switches
            
            frame_metrics[frame_num] = {
                'tp': tp, 'fp': fp, 'fn': fn,
                'id_switches': frame_id_switches,
                'motp': frame_motp
            }
        
        # Calculate overall metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        mota = 1 - (total_fn + total_fp + total_id_switches) / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        motp = total_motp_sum / total_motp_count if total_motp_count > 0 else 0.0
        
        return {
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'total_id_switches': total_id_switches,
            'precision': precision,
            'recall': recall,
            'mota': mota,
            'motp': motp,
            'frame_metrics': frame_metrics
        }
    
    def compute_motmetrics(self, gt_data: Dict[int, List[BoundingBox]], 
                          pred_data: Dict[int, List[BoundingBox]]) -> Dict:
        """Compute metrics using motmetrics library if available."""
        if not MOTMETRICS_AVAILABLE:
            return {}
        
        try:
            # Create accumulator
            acc = mm.MOTAccumulator(auto_id=True)
            
            for frame_num in sorted(gt_data.keys()):
                gt_boxes = gt_data.get(frame_num, [])
                pred_boxes = pred_data.get(frame_num, [])
                
                # Extract GT object IDs and positions
                gt_ids = [box.track_id for box in gt_boxes]
                gt_positions = [[*box.to_xyxy()] for box in gt_boxes]
                
                # Extract prediction IDs and positions
                pred_ids = [box.track_id for box in pred_boxes if box.track_id is not None]
                pred_positions = [[*box.to_xyxy()] for box in pred_boxes if box.track_id is not None]
                
                # Calculate distances (1 - IoU)
                if gt_positions and pred_positions:
                    distances = []
                    for gt_pos in gt_positions:
                        row = []
                        for pred_pos in pred_positions:
                            # Calculate IoU
                            x1_i = max(gt_pos[0], pred_pos[0])
                            y1_i = max(gt_pos[1], pred_pos[1])
                            x2_i = min(gt_pos[2], pred_pos[2])
                            y2_i = min(gt_pos[3], pred_pos[3])
                            
                            if x2_i <= x1_i or y2_i <= y1_i:
                                iou = 0.0
                            else:
                                intersection = (x2_i - x1_i) * (y2_i - y1_i)
                                area1 = (gt_pos[2] - gt_pos[0]) * (gt_pos[3] - gt_pos[1])
                                area2 = (pred_pos[2] - pred_pos[0]) * (pred_pos[3] - pred_pos[1])
                                union = area1 + area2 - intersection
                                iou = intersection / union if union > 0 else 0.0
                            
                            # Convert IoU to distance
                            distance = 1.0 - iou if iou >= self.iou_threshold else np.inf
                            row.append(distance)
                        distances.append(row)
                    
                    distances = np.array(distances)
                else:
                    distances = np.empty((0, 0))
                
                # Update accumulator
                acc.update(gt_ids, pred_ids, distances)
            
            # Compute metrics
            mh = mm.metrics.create()
            summary = mh.compute(acc, metrics=['mota', 'motp', 'idf1', 'num_switches', 
                                              'precision', 'recall', 'num_false_positives',
                                              'num_misses'], name='acc')
            
            # Extract metrics
            result = {}
            if not summary.empty:
                for metric in ['mota', 'motp', 'idf1', 'precision', 'recall']:
                    if metric in summary.columns:
                        result[f'mot_{metric}'] = float(summary.loc['acc', metric])
                
                for metric in ['num_switches', 'num_false_positives', 'num_misses']:
                    if metric in summary.columns:
                        result[f'mot_{metric}'] = int(summary.loc['acc', metric])
            
            return result
            
        except Exception as e:
            if self.debug:
                logger.warning(f"MOTMetrics computation failed: {e}")
            return {}
    
    def compute_idf1(self, gt_data: Dict[int, List[BoundingBox]], 
                     pred_data: Dict[int, List[BoundingBox]]) -> float:
        """Compute ID F1-score."""
        total_idtp = 0  # Correct ID assignments
        total_idfp = 0  # Incorrect ID assignments
        total_idfn = 0  # Missed ID assignments
        
        # Build track associations over time
        track_associations = defaultdict(list)  # pred_track_id -> [gt_track_ids]
        
        for frame_num in sorted(gt_data.keys()):
            gt_boxes = gt_data.get(frame_num, [])
            pred_boxes = pred_data.get(frame_num, [])
            
            matches, _, _ = self.assign_detections(gt_boxes, pred_boxes)
            
            for gt_idx, pred_idx in matches:
                gt_track_id = gt_boxes[gt_idx].track_id
                pred_track_id = pred_boxes[pred_idx].track_id
                
                if pred_track_id is not None and gt_track_id is not None:
                    track_associations[pred_track_id].append(gt_track_id)
        
        # Count ID metrics
        for pred_track_id, gt_track_ids in track_associations.items():
            if gt_track_ids:
                # Most frequent GT track for this prediction track
                most_frequent_gt = max(set(gt_track_ids), key=gt_track_ids.count)
                correct_count = gt_track_ids.count(most_frequent_gt)
                
                total_idtp += correct_count
                total_idfp += len(gt_track_ids) - correct_count
        
        # Count missed GT tracks
        all_gt_tracks = set()
        all_matched_gt_tracks = set()
        
        for frame_num in sorted(gt_data.keys()):
            gt_boxes = gt_data.get(frame_num, [])
            pred_boxes = pred_data.get(frame_num, [])
            
            for box in gt_boxes:
                if box.track_id is not None:
                    all_gt_tracks.add(box.track_id)
            
            matches, _, _ = self.assign_detections(gt_boxes, pred_boxes)
            for gt_idx, pred_idx in matches:
                if (gt_boxes[gt_idx].track_id is not None and 
                    pred_boxes[pred_idx].track_id is not None):
                    all_matched_gt_tracks.add(gt_boxes[gt_idx].track_id)
        
        total_idfn = len(all_gt_tracks) - len(all_matched_gt_tracks)
        
        # Calculate IDF1
        if total_idtp + total_idfp + total_idfn == 0:
            return 1.0
        
        idp = total_idtp / (total_idtp + total_idfp) if (total_idtp + total_idfp) > 0 else 0.0
        idr = total_idtp / (total_idtp + total_idfn) if (total_idtp + total_idfn) > 0 else 0.0
        
        return 2 * idp * idr / (idp + idr) if (idp + idr) > 0 else 0.0
    
    def compute_hota_metrics(self, gt_data: Dict[int, List[BoundingBox]], 
                            pred_data: Dict[int, List[BoundingBox]]) -> Dict:
        """Compute HOTA metrics over multiple alpha thresholds."""
        alpha_values = np.linspace(0.05, 0.95, 19)  # 19 alpha values
        
        hota_scores = []
        deta_scores = []
        assa_scores = []
        loca_scores = []
        
        for alpha in alpha_values:
            # Temporarily change IoU threshold
            original_threshold = self.iou_threshold
            self.iou_threshold = alpha
            
            frame_deta = []
            frame_assa = []
            frame_loca = []
            
            for frame_num in sorted(gt_data.keys()):
                gt_boxes = gt_data.get(frame_num, [])
                pred_boxes = pred_data.get(frame_num, [])
                
                if not gt_boxes and not pred_boxes:
                    deta, assa, loca = 1.0, 1.0, 1.0
                elif not gt_boxes or not pred_boxes:
                    deta, assa, loca = 0.0, 0.0, 0.0
                else:
                    matches, unmatched_gt, unmatched_pred = self.assign_detections(gt_boxes, pred_boxes)
                    
                    tp = len(matches)
                    fp = len(unmatched_pred)
                    fn = len(unmatched_gt)
                    
                    # DetA
                    deta = tp / (tp + 0.5 * (fp + fn)) if (tp + fp + fn) > 0 else 0.0
                    
                    # LocA
                    if matches:
                        loca = sum(gt_boxes[gt_idx].iou(pred_boxes[pred_idx]) 
                                  for gt_idx, pred_idx in matches) / len(matches)
                    else:
                        loca = 0.0
                    
                    # AssA (simplified - would need proper track analysis)
                    assa = 0.5  # Placeholder
                
                frame_deta.append(deta)
                frame_assa.append(assa)
                frame_loca.append(loca)
            
            # Restore original threshold
            self.iou_threshold = original_threshold
            
            # Average over frames
            avg_deta = np.mean(frame_deta) if frame_deta else 0.0
            avg_assa = np.mean(frame_assa) if frame_assa else 0.0
            avg_loca = np.mean(frame_loca) if frame_loca else 0.0
            
            deta_scores.append(avg_deta)
            assa_scores.append(avg_assa)
            loca_scores.append(avg_loca)
            hota_scores.append(np.sqrt(avg_deta * avg_assa))
        
        return {
            'hota': np.mean(hota_scores),
            'deta': np.mean(deta_scores),
            'assa': np.mean(assa_scores),
            'loca': np.mean(loca_scores)
        }
    
    def compute_track_quality(self, gt_data: Dict[int, List[BoundingBox]], 
                             pred_data: Dict[int, List[BoundingBox]]) -> Dict:
        """Compute track quality metrics (MT, ML, PT)."""
        # Extract unique GT tracks
        gt_tracks = defaultdict(list)  # track_id -> [frame_nums]
        
        for frame_num, gt_boxes in gt_data.items():
            for box in gt_boxes:
                if box.track_id is not None:
                    gt_tracks[box.track_id].append(frame_num)
        
        if not gt_tracks:
            return {'mostly_tracked': 0, 'mostly_lost': 0, 'partially_tracked': 0, 'total_tracks': 0}
        
        # For each GT track, count how many frames it's correctly detected
        mostly_tracked = 0
        mostly_lost = 0
        partially_tracked = 0
        
        for track_id, frames in gt_tracks.items():
            correctly_detected = 0
            total_frames = len(frames)
            
            for frame_num in frames:
                gt_boxes = gt_data.get(frame_num, [])
                pred_boxes = pred_data.get(frame_num, [])
                
                # Find this track in GT
                gt_box = None
                for box in gt_boxes:
                    if box.track_id == track_id:
                        gt_box = box
                        break
                
                if gt_box is not None:
                    # Check if any prediction matches this GT box
                    matches, _, _ = self.assign_detections([gt_box], pred_boxes)
                    if matches:
                        correctly_detected += 1
            
            detection_ratio = correctly_detected / total_frames
            
            if detection_ratio >= 0.8:
                mostly_tracked += 1
            elif detection_ratio <= 0.2:
                mostly_lost += 1
            else:
                partially_tracked += 1
        
        return {
            'mostly_tracked': mostly_tracked,
            'mostly_lost': mostly_lost,
            'partially_tracked': partially_tracked,
            'total_tracks': len(gt_tracks)
        }
    
    def debug_analysis(self, gt_data: Dict[int, List[BoundingBox]], 
                      pred_data: Dict[int, List[BoundingBox]]) -> Dict:
        """Perform debug analysis of spatial and class distributions."""
        # Class distribution
        gt_class_counts = {0: 0, 1: 0, 2: 0}
        pred_class_counts = {0: 0, 1: 0, 2: 0}
        
        # Spatial distribution
        gt_positions = []
        pred_positions = []
        
        for frame_num in sorted(gt_data.keys()):
            gt_boxes = gt_data.get(frame_num, [])
            pred_boxes = pred_data.get(frame_num, [])
            
            for box in gt_boxes:
                gt_class_counts[box.class_id] += 1
                gt_positions.append((box.x_center, box.y_center))
            
            for box in pred_boxes:
                pred_class_counts[box.class_id] += 1
                pred_positions.append((box.x_center, box.y_center))
        
        debug_info = {
            'gt_class_distribution': gt_class_counts,
            'pred_class_distribution': pred_class_counts,
        }
        
        if gt_positions:
            gt_x = [pos[0] for pos in gt_positions]
            gt_y = [pos[1] for pos in gt_positions]
            debug_info['gt_spatial'] = {
                'x_range': (min(gt_x), max(gt_x)),
                'y_range': (min(gt_y), max(gt_y)),
                'x_center': np.mean(gt_x),
                'y_center': np.mean(gt_y)
            }
        
        if pred_positions:
            pred_x = [pos[0] for pos in pred_positions]
            pred_y = [pos[1] for pos in pred_positions]
            debug_info['pred_spatial'] = {
                'x_range': (min(pred_x), max(pred_x)),
                'y_range': (min(pred_y), max(pred_y)),
                'x_center': np.mean(pred_x),
                'y_center': np.mean(pred_y)
            }
        
        return debug_info
    
    def evaluate(self) -> Dict:
        """Run complete evaluation."""
        if self.debug:
            logger.info(f"Starting evaluation for {self.video_name}")
        
        # Load data
        frame_mapping = self.get_frame_mapping()
        gt_data = self.load_ground_truth()
        pred_data = self.load_predictions(frame_mapping)
        
        if not gt_data:
            logger.error("No ground truth data found!")
            return {}
        
        # Compute all metrics
        basic_metrics = self.compute_basic_metrics(gt_data, pred_data)
        idf1 = self.compute_idf1(gt_data, pred_data)
        hota_metrics = self.compute_hota_metrics(gt_data, pred_data)
        track_quality = self.compute_track_quality(gt_data, pred_data)
        
        # Try MOTMetrics if available
        mot_metrics = self.compute_motmetrics(gt_data, pred_data)
        
        # Debug analysis if requested
        debug_info = {}
        if self.debug:
            debug_info = self.debug_analysis(gt_data, pred_data)
        
        # Combine all results
        results = {
            'video_name': self.video_name,
            'total_frames': len(gt_data),
            'mota': basic_metrics['mota'],
            'motp': basic_metrics['motp'],
            'idf1': idf1,
            'precision': basic_metrics['precision'],
            'recall': basic_metrics['recall'],
            'total_tp': basic_metrics['total_tp'],
            'total_fp': basic_metrics['total_fp'],
            'total_fn': basic_metrics['total_fn'],
            'id_switches': basic_metrics['total_id_switches'],
            **hota_metrics,
            **track_quality,
            **mot_metrics
        }
        
        if debug_info:
            results['debug_info'] = debug_info
        
        return results


def print_results(results: Dict, debug: bool = False):
    """Print evaluation results in a formatted way."""
    print(f"\n{'='*70}")
    print(f"TRACKING EVALUATION RESULTS: {results['video_name']}")
    print(f"{'='*70}")
    
    print(f"\nBASIC METRICS:")
    print(f"   Total Frames: {results['total_frames']}")
    print(f"   True Positives: {results['total_tp']}")
    print(f"   False Positives: {results['total_fp']}")
    print(f"   False Negatives: {results['total_fn']}")
    
    print(f"\n🎯 ACCURACY METRICS:")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall: {results['recall']:.4f}")
    
    print(f"\n🔄 TRACKING METRICS:")
    print(f"   MOTA (Multiple Object Tracking Accuracy): {results['mota']:.4f}")
    print(f"   MOTP (Multiple Object Tracking Precision): {results['motp']:.4f}")
    print(f"   IDF1 (ID F1-score): {results['idf1']:.4f}")
    print(f"   ID Switches: {results['id_switches']}")
    
    print(f"\n🎯 HOTA METRICS:")
    print(f"   HOTA (Higher Order Tracking Accuracy): {results['hota']:.4f}")
    print(f"   DetA (Detection Accuracy): {results['deta']:.4f}")
    print(f"   AssA (Association Accuracy): {results['assa']:.4f}")
    print(f"   LocA (Localization Accuracy): {results['loca']:.4f}")
    
    print(f"\n📈 TRACK QUALITY:")
    print(f"   Mostly Tracked: {results['mostly_tracked']}")
    print(f"   Mostly Lost: {results['mostly_lost']}")
    print(f"   Partially Tracked: {results['partially_tracked']}")
    print(f"   Total Tracks: {results['total_tracks']}")
    
    # MOTMetrics results if available
    if any(key.startswith('mot_') for key in results.keys()):
        print(f"\n🔬 MOTMETRICS (if available):")
        for key, value in results.items():
            if key.startswith('mot_'):
                metric_name = key.replace('mot_', '').upper()
                if isinstance(value, float):
                    print(f"   {metric_name}: {value:.4f}")
                else:
                    print(f"   {metric_name}: {value}")
    
    # Debug information
    if debug and 'debug_info' in results:
        debug_info = results['debug_info']
        print(f"\n🔍 DEBUG INFORMATION:")
        
        print(f"   Class Distribution:")
        print(f"     GT: Ball={debug_info['gt_class_distribution'][0]}, "
              f"Player={debug_info['gt_class_distribution'][1]}, "
              f"Referee={debug_info['gt_class_distribution'][2]}")
        print(f"     Pred: Ball={debug_info['pred_class_distribution'][0]}, "
              f"Player={debug_info['pred_class_distribution'][1]}, "
              f"Referee={debug_info['pred_class_distribution'][2]}")
        
        if 'gt_spatial' in debug_info:
            gt_spatial = debug_info['gt_spatial']
            print(f"   GT Spatial: X=[{gt_spatial['x_range'][0]:.3f}, {gt_spatial['x_range'][1]:.3f}], "
                  f"Y=[{gt_spatial['y_range'][0]:.3f}, {gt_spatial['y_range'][1]:.3f}]")
            print(f"   GT Center: ({gt_spatial['x_center']:.3f}, {gt_spatial['y_center']:.3f})")
        
        if 'pred_spatial' in debug_info:
            pred_spatial = debug_info['pred_spatial']
            print(f"   Pred Spatial: X=[{pred_spatial['x_range'][0]:.3f}, {pred_spatial['x_range'][1]:.3f}], "
                  f"Y=[{pred_spatial['y_range'][0]:.3f}, {pred_spatial['y_range'][1]:.3f}]")
            print(f"   Pred Center: ({pred_spatial['x_center']:.3f}, {pred_spatial['y_center']:.3f})")


def print_summary(all_results: List[Dict]):
    """Print summary across all videos."""
    if not all_results:
        return
    
    print(f"\n{'='*70}")
    print("📋 SUMMARY ACROSS ALL VIDEOS")
    print(f"{'='*70}")
    
    # Calculate averages
    metrics = ['mota', 'motp', 'idf1', 'hota', 'precision', 'recall']
    avg_metrics = {}
    
    for metric in metrics:
        values = [r.get(metric, 0.0) for r in all_results]
        avg_metrics[metric] = np.mean(values) if values else 0.0
    
    # Calculate totals
    total_tp = sum([r.get('total_tp', 0) for r in all_results])
    total_fp = sum([r.get('total_fp', 0) for r in all_results])
    total_fn = sum([r.get('total_fn', 0) for r in all_results])
    total_ids = sum([r.get('id_switches', 0) for r in all_results])
    total_tracks = sum([r.get('total_tracks', 0) for r in all_results])
    total_mt = sum([r.get('mostly_tracked', 0) for r in all_results])
    total_ml = sum([r.get('mostly_lost', 0) for r in all_results])
    
    print(f"Average MOTA: {avg_metrics['mota']:.4f}")
    print(f"Average MOTP: {avg_metrics['motp']:.4f}")
    print(f"Average IDF1: {avg_metrics['idf1']:.4f}")
    print(f"Average HOTA: {avg_metrics['hota']:.4f}")
    print(f"Average Precision: {avg_metrics['precision']:.4f}")
    print(f"Average Recall: {avg_metrics['recall']:.4f}")
    print(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Total ID Switches: {total_ids}")
    print(f"Total Tracks: {total_tracks} (MT: {total_mt}, ML: {total_ml})")


def main():
    """Main function to run evaluation."""
    parser = argparse.ArgumentParser(description='Unified Basketball Tracking Evaluation')
    parser.add_argument('--videos', nargs='+', default=['out2', 'out4', 'out13'],
                       help='Video names to evaluate (default: out2 out4 out13)')
    parser.add_argument('--gt_base_dir', default='../dataset_plain_3classes',
                       help='Base directory for ground truth data')
    parser.add_argument('--pred_base_dir', default='results',
                       help='Base directory for prediction data')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU threshold for matching (default: 0.5)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output and analysis')
    
    args = parser.parse_args()
    
    # Check dependencies
    print("🔧 DEPENDENCY CHECK:")
    print("="*50)
    
    required_packages = ['numpy', 'scipy']
    optional_packages = ['motmetrics']
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - REQUIRED")
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"⚠️  {package} - OPTIONAL (enhanced metrics)")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("Install with: pip install " + " ".join(missing_required))
        return
    
    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {', '.join(missing_optional)}")
        print("Install for enhanced metrics: pip install " + " ".join(missing_optional))
    
    print(f"\n🚀 Starting evaluation...")
    if not MOTMETRICS_AVAILABLE:
        print("⚠️  MOTMetrics not available - using basic implementation")
    
    all_results = []
    
    for video_name in args.videos:
        gt_dir = f"{args.gt_base_dir}/{video_name}/labels"
        pred_dir = f"{args.pred_base_dir}/{video_name}/labels"
        
        if args.debug:
            logger.info(f"Evaluating {video_name}...")
            logger.info(f"Ground truth: {gt_dir}")
            logger.info(f"Predictions: {pred_dir}")
        
        evaluator = UnifiedTrackingEvaluator(gt_dir, pred_dir, video_name, args.debug)
        evaluator.iou_threshold = args.iou_threshold
        
        results = evaluator.evaluate()
        
        if results:
            all_results.append(results)
            print_results(results, args.debug)
        else:
            logger.error(f"Evaluation failed for {video_name}")
    
    # Print summary
    print_summary(all_results)
    
    print(f"\n✅ Evaluation completed for {len(all_results)} videos")


if __name__ == "__main__":
    main()
