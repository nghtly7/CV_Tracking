"""
Diagnostic Basketball Tracking Evaluation
=========================================

This version forces matches to show what metrics would look like and provides detailed debugging.
"""

import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from collections import defaultdict
import logging

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
        area1 = self.width * self.height
        area2 = other.width * other.height
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def euclidean_distance(self, other: 'BoundingBox') -> float:
        """Calculate center-to-center Euclidean distance."""
        dx = self.x_center - other.x_center
        dy = self.y_center - other.y_center
        return np.sqrt(dx*dx + dy*dy)


class DiagnosticEvaluator:
    """Diagnostic evaluator that forces matches and shows detailed info."""
    
    def __init__(self, gt_dir: str, pred_dir: str, video_name: str):
        self.gt_dir = Path(gt_dir)
        self.pred_dir = Path(pred_dir)
        self.video_name = video_name
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
                        
                        # Check if track ID is present
                        track_id = None
                        if len(parts) >= 6:
                            try:
                                track_id = int(float(parts[5]))
                            except:
                                track_id = None
                        
                        # Validate coordinates
                        if (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and
                            0.0 <= width <= 1.0 and 0.0 <= height <= 1.0):
                            boxes.append(BoundingBox(x_center, y_center, width, height, 
                                                   class_id, track_id))
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
        
        return boxes
    
    def get_frame_mapping(self) -> Dict[int, int]:
        """Create mapping from GT frame numbers to tracking frame numbers."""
        mapping = {}
        for gt_frame in range(1, 121):  # GT frames 1-120
            tracking_frame = (gt_frame - 1) * 5 + 3  # 3, 8, 13, ..., 598
            mapping[gt_frame] = tracking_frame
        return mapping
    
    def load_data(self) -> Tuple[Dict, Dict]:
        """Load GT and prediction data."""
        # Load GT
        gt_data = {}
        gt_files = sorted(self.gt_dir.glob("*.txt"))
        
        for i, gt_file in enumerate(gt_files, 1):
            if i <= 120:
                boxes = self.load_yolo_labels(gt_file)
                for j, box in enumerate(boxes):
                    if box.track_id is None:
                        box.track_id = f"gt_{i}_{j}"
                gt_data[i] = boxes
        
        # Load predictions
        frame_mapping = self.get_frame_mapping()
        pred_data = {}
        
        for gt_frame, tracking_frame in frame_mapping.items():
            pred_file = self.pred_dir / f"frame_{tracking_frame:06d}.txt"
            if pred_file.exists():
                boxes = self.load_yolo_labels(pred_file)
                pred_data[gt_frame] = boxes
            else:
                pred_data[gt_frame] = []
        
        return gt_data, pred_data
    
    def detailed_frame_analysis(self, gt_data: Dict, pred_data: Dict, num_frames: int = 5):
        """Analyze first few frames in detail."""
        print(f"\n🔍 DETAILED FRAME ANALYSIS (First {num_frames} frames):")
        print("="*70)
        
        for frame_num in sorted(list(gt_data.keys())[:num_frames]):
            gt_boxes = gt_data.get(frame_num, [])
            pred_boxes = pred_data.get(frame_num, [])
            
            print(f"\n--- FRAME {frame_num} ---")
            print(f"GT boxes: {len(gt_boxes)}, Pred boxes: {len(pred_boxes)}")
            
            # Show GT boxes
            for i, box in enumerate(gt_boxes):
                print(f"  GT{i+1}: class={box.class_id}({self.class_names[box.class_id]}), "
                      f"center=({box.x_center:.3f},{box.y_center:.3f}), "
                      f"size=({box.width:.3f}x{box.height:.3f})")
            
            # Show Pred boxes
            for i, box in enumerate(pred_boxes):
                print(f"  Pred{i+1}: class={box.class_id}({self.class_names[box.class_id]}), "
                      f"center=({box.x_center:.3f},{box.y_center:.3f}), "
                      f"size=({box.width:.3f}x{box.height:.3f})")
            
            # Calculate all pairwise IoUs and distances
            if gt_boxes and pred_boxes:
                print(f"  Pairwise Analysis:")
                for i, gt_box in enumerate(gt_boxes):
                    for j, pred_box in enumerate(pred_boxes):
                        iou = gt_box.iou(pred_box)
                        dist = gt_box.euclidean_distance(pred_box)
                        same_class = gt_box.class_id == pred_box.class_id
                        print(f"    GT{i+1} vs Pred{j+1}: IoU={iou:.4f}, dist={dist:.3f}, same_class={same_class}")
    
    def forced_matching_analysis(self, gt_data: Dict, pred_data: Dict):
        """Force matches using different strategies and show results."""
        print(f"\n🎯 FORCED MATCHING ANALYSIS:")
        print("="*70)
        
        strategies = [
            ("IoU >= 0.5 + Same Class", lambda gt, pred: gt.iou(pred) >= 0.5 and gt.class_id == pred.class_id),
            ("IoU >= 0.1 + Same Class", lambda gt, pred: gt.iou(pred) >= 0.1 and gt.class_id == pred.class_id), 
            ("Any IoU > 0 + Same Class", lambda gt, pred: gt.iou(pred) > 0 and gt.class_id == pred.class_id),
            ("IoU >= 0.5 + Any Class", lambda gt, pred: gt.iou(pred) >= 0.5),
            ("Distance < 0.1 + Same Class", lambda gt, pred: gt.euclidean_distance(pred) < 0.1 and gt.class_id == pred.class_id),
            ("Distance < 0.2 + Same Class", lambda gt, pred: gt.euclidean_distance(pred) < 0.2 and gt.class_id == pred.class_id),
            ("Distance < 0.5 + Same Class", lambda gt, pred: gt.euclidean_distance(pred) < 0.5 and gt.class_id == pred.class_id),
            ("Closest Distance + Same Class", lambda gt, pred: gt.class_id == pred.class_id),  # Will use closest
        ]
        
        for strategy_name, match_func in strategies:
            total_matches = 0
            total_gt = 0
            total_pred = 0
            
            for frame_num in sorted(gt_data.keys())[:10]:  # First 10 frames
                gt_boxes = gt_data.get(frame_num, [])
                pred_boxes = pred_data.get(frame_num, [])
                
                total_gt += len(gt_boxes)
                total_pred += len(pred_boxes)
                
                # Try to match with this strategy
                matches = []
                used_gt = set()
                used_pred = set()
                
                if strategy_name == "Closest Distance + Same Class":
                    # Special case: match by closest distance
                    for i, gt_box in enumerate(gt_boxes):
                        best_match = None
                        best_distance = float('inf')
                        
                        for j, pred_box in enumerate(pred_boxes):
                            if j in used_pred:
                                continue
                            if gt_box.class_id == pred_box.class_id:
                                dist = gt_box.euclidean_distance(pred_box)
                                if dist < best_distance:
                                    best_distance = dist
                                    best_match = j
                        
                        if best_match is not None:
                            matches.append((i, best_match))
                            used_gt.add(i)
                            used_pred.add(best_match)
                else:
                    # Regular matching
                    for i, gt_box in enumerate(gt_boxes):
                        for j, pred_box in enumerate(pred_boxes):
                            if i in used_gt or j in used_pred:
                                continue
                            if match_func(gt_box, pred_box):
                                matches.append((i, j))
                                used_gt.add(i)
                                used_pred.add(j)
                                break
                
                total_matches += len(matches)
            
            tp = total_matches
            fp = total_pred - total_matches
            fn = total_gt - total_matches
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            mota = 1 - (fn + fp) / (tp + fn) if (tp + fn) > 0 else 0.0
            
            print(f"\n{strategy_name}:")
            print(f"  TP={tp}, FP={fp}, FN={fn}")
            print(f"  Precision={precision:.3f}, Recall={recall:.3f}, MOTA={mota:.3f}")
    
    def spatial_distribution_analysis(self, gt_data: Dict, pred_data: Dict):
        """Analyze spatial distributions."""
        print(f"\n📍 SPATIAL DISTRIBUTION ANALYSIS:")
        print("="*70)
        
        # Collect all positions
        gt_positions_by_class = {0: [], 1: [], 2: []}
        pred_positions_by_class = {0: [], 1: [], 2: []}
        
        for frame_num in sorted(gt_data.keys()):
            gt_boxes = gt_data.get(frame_num, [])
            pred_boxes = pred_data.get(frame_num, [])
            
            for box in gt_boxes:
                gt_positions_by_class[box.class_id].append((box.x_center, box.y_center))
            
            for box in pred_boxes:
                pred_positions_by_class[box.class_id].append((box.x_center, box.y_center))
        
        # Analyze by class
        for class_id in [0, 1, 2]:
            class_name = self.class_names[class_id]
            gt_pos = gt_positions_by_class[class_id]
            pred_pos = pred_positions_by_class[class_id]
            
            print(f"\n{class_name} (Class {class_id}):")
            
            if gt_pos:
                gt_x = [p[0] for p in gt_pos]
                gt_y = [p[1] for p in gt_pos]
                print(f"  GT: {len(gt_pos)} instances")
                print(f"      X: [{min(gt_x):.3f}, {max(gt_x):.3f}] mean={np.mean(gt_x):.3f}")
                print(f"      Y: [{min(gt_y):.3f}, {max(gt_y):.3f}] mean={np.mean(gt_y):.3f}")
            else:
                print(f"  GT: 0 instances")
            
            if pred_pos:
                pred_x = [p[0] for p in pred_pos]
                pred_y = [p[1] for p in pred_pos]
                print(f"  Pred: {len(pred_pos)} instances")
                print(f"        X: [{min(pred_x):.3f}, {max(pred_x):.3f}] mean={np.mean(pred_x):.3f}")
                print(f"        Y: [{min(pred_y):.3f}, {max(pred_y):.3f}] mean={np.mean(pred_y):.3f}")
                
                if gt_pos:
                    # Calculate separation
                    x_sep = abs(np.mean(gt_x) - np.mean(pred_x))
                    y_sep = abs(np.mean(gt_y) - np.mean(pred_y))
                    print(f"        Separation: X={x_sep:.3f}, Y={y_sep:.3f}")
            else:
                print(f"  Pred: 0 instances")
    
    def simulated_metrics_with_forced_matches(self, gt_data: Dict, pred_data: Dict):
        """Simulate what metrics would look like with forced closest matches."""
        print(f"\n🧪 SIMULATED METRICS (Forced Closest Matches by Class):")
        print("="*70)
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_motp_sum = 0.0
        total_motp_count = 0
        
        for frame_num in sorted(gt_data.keys()):
            gt_boxes = gt_data.get(frame_num, [])
            pred_boxes = pred_data.get(frame_num, [])
            
            # Force matches by finding closest same-class pairs
            matches = []
            used_pred = set()
            
            for i, gt_box in enumerate(gt_boxes):
                best_match = None
                best_distance = float('inf')
                
                for j, pred_box in enumerate(pred_boxes):
                    if j in used_pred:
                        continue
                    if gt_box.class_id == pred_box.class_id:
                        dist = gt_box.euclidean_distance(pred_box)
                        if dist < best_distance:
                            best_distance = dist
                            best_match = j
                
                if best_match is not None:
                    matches.append((i, best_match))
                    used_pred.add(best_match)
            
            tp = len(matches)
            fp = len(pred_boxes) - tp
            fn = len(gt_boxes) - tp
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            # Calculate MOTP using distance instead of IoU
            if matches:
                motp_sum = 0.0
                for gt_idx, pred_idx in matches:
                    distance = gt_boxes[gt_idx].euclidean_distance(pred_boxes[pred_idx])
                    # Convert distance to similarity (1 - normalized_distance)
                    similarity = max(0, 1 - distance)  # Higher is better
                    motp_sum += similarity
                
                frame_motp = motp_sum / len(matches)
                total_motp_sum += motp_sum
                total_motp_count += len(matches)
        
        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        mota = 1 - (total_fn + total_fp) / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        motp = total_motp_sum / total_motp_count if total_motp_count > 0 else 0.0
        
        print(f"Forced Matching Results:")
        print(f"  Total TP: {total_tp}")
        print(f"  Total FP: {total_fp}")
        print(f"  Total FN: {total_fn}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  MOTA: {mota:.4f}")
        print(f"  MOTP (distance-based): {motp:.4f}")
        
        print(f"\n💡 This shows what your metrics WOULD BE if tracking worked correctly!")
    
    def run_diagnosis(self):
        """Run complete diagnostic analysis."""
        print(f"🔬 DIAGNOSTIC ANALYSIS: {self.video_name}")
        print("="*70)
        
        # Load data
        gt_data, pred_data = self.load_data()
        
        if not gt_data:
            print("❌ No ground truth data found!")
            return
        
        print(f"✅ Loaded GT: {len(gt_data)} frames, Pred: {len(pred_data)} frames")
        
        # Run all analyses
        self.detailed_frame_analysis(gt_data, pred_data)
        self.spatial_distribution_analysis(gt_data, pred_data)
        self.forced_matching_analysis(gt_data, pred_data)
        self.simulated_metrics_with_forced_matches(gt_data, pred_data)
        
        print(f"\n🎯 CONCLUSIONS:")
        print("="*70)
        print("1. If TP is still 0 after this analysis, the issue is:")
        print("   - Complete spatial separation (different image regions)")
        print("   - OR wrong class predictions") 
        print("   - OR frame alignment issues")
        print()
        print("2. The 'Simulated Metrics' show what you'd get with working tracking")
        print()
        print("3. Check the tracking video to see if objects are being detected at all")


def main():
    parser = argparse.ArgumentParser(description='Diagnostic Basketball Tracking Analysis')
    parser.add_argument('--videos', nargs='+', default=['out2'],
                       help='Video names to analyze')
    parser.add_argument('--gt_base_dir', default='../dataset_plain_3classes',
                       help='Base directory for ground truth data')
    parser.add_argument('--pred_base_dir', default='results',
                       help='Base directory for prediction data')
    
    args = parser.parse_args()
    
    for video_name in args.videos:
        gt_dir = f"{args.gt_base_dir}/{video_name}/labels"
        pred_dir = f"{args.pred_base_dir}/{video_name}/labels"
        
        evaluator = DiagnosticEvaluator(gt_dir, pred_dir, video_name)
        evaluator.run_diagnosis()
        
        if len(args.videos) > 1:
            print("\n" + "="*100 + "\n")

if __name__ == "__main__":
    main()
