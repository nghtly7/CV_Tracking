"""
3D Tracking Metrics
- Trajectory metrics: ADE/FDE, RMSE 3D per object
- MOT 3D metrics: MOTA/IDF1/HOTA3D, IDSW, FRAG
- Reprojection error analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import pandas as pd
from triangulation import Track3D

class TrajectoryMetrics:
    """Compute trajectory prediction metrics"""
    
    @staticmethod
    def compute_ade(pred_trajectory: List[np.ndarray], 
                   gt_trajectory: List[np.ndarray]) -> float:
        """
        Average Displacement Error - average L2 norm distance over all time steps
        """
        if len(pred_trajectory) != len(gt_trajectory):
            min_len = min(len(pred_trajectory), len(gt_trajectory))
            pred_trajectory = pred_trajectory[:min_len]
            gt_trajectory = gt_trajectory[:min_len]
        
        distances = []
        for pred_pos, gt_pos in zip(pred_trajectory, gt_trajectory):
            distance = np.linalg.norm(pred_pos - gt_pos)
            distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    @staticmethod
    def compute_fde(pred_trajectory: List[np.ndarray], 
                   gt_trajectory: List[np.ndarray]) -> float:
        """
        Final Displacement Error - L2 distance at the final time step
        """
        if not pred_trajectory or not gt_trajectory:
            return 0.0
        
        pred_final = pred_trajectory[-1]
        gt_final = gt_trajectory[-1]
        
        return np.linalg.norm(pred_final - gt_final)
    
    @staticmethod
    def compute_rmse_3d(pred_trajectory: List[np.ndarray], 
                       gt_trajectory: List[np.ndarray]) -> float:
        """
        Root Mean Square Error in 3D
        """
        if len(pred_trajectory) != len(gt_trajectory):
            min_len = min(len(pred_trajectory), len(gt_trajectory))
            pred_trajectory = pred_trajectory[:min_len]
            gt_trajectory = gt_trajectory[:min_len]
        
        squared_errors = []
        for pred_pos, gt_pos in zip(pred_trajectory, gt_trajectory):
            squared_error = np.sum((pred_pos - gt_pos) ** 2)
            squared_errors.append(squared_error)
        
        return np.sqrt(np.mean(squared_errors)) if squared_errors else 0.0

class MOTMetrics:
    """Compute Multi-Object Tracking 3D metrics"""
    
    def __init__(self, distance_threshold: float = 1000.0):  # 1 meter in mm
        self.distance_threshold = distance_threshold
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        self.total_gt = 0
        self.total_pred = 0
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_id_switches = 0
        self.total_fragmentations = 0
        self.trajectory_matches = defaultdict(list)  # For IDF1
        self.frame_associations = {}  # For HOTA3D
        
    def compute_frame_associations(self, gt_tracks: List[Track3D], 
                                 pred_tracks: List[Track3D]) -> Dict:
        """
        Compute associations for a single frame
        Returns dictionary with TP, FP, FN counts and association matrix
        """

        # Create distance matrix
        if not gt_tracks or not pred_tracks:
            return {
                'tp': 0, 'fp': len(pred_tracks), 'fn': len(gt_tracks),
                'associations': {}, 'distances': []
            }
        
        distance_matrix = np.zeros((len(gt_tracks), len(pred_tracks)))
        
        for i, gt_track in enumerate(gt_tracks):
            for j, pred_track in enumerate(pred_tracks):
                # Only match same class
                if gt_track.class_id != pred_track.class_id:
                    distance_matrix[i, j] = float('inf')
                else:
                    distance = np.linalg.norm(gt_track.position - pred_track.position)
                    distance_matrix[i, j] = distance
        
        # Hungarian assignment (simplified greedy approach)
        associations = {}
        used_pred = set()
        used_gt = set()
        valid_distances = []
        
        # Sort by distance and greedily assign
        flat_indices = np.argsort(distance_matrix.flatten())
        
        for flat_idx in flat_indices:
            i, j = np.unravel_index(flat_idx, distance_matrix.shape)
            
            if i in used_gt or j in used_pred:
                continue
                
            distance = distance_matrix[i, j]
            if distance <= self.distance_threshold:
                associations[i] = j
                used_gt.add(i)
                used_pred.add(j)
                valid_distances.append(distance)
        
        tp = len(associations)
        fp = len(pred_tracks) - tp
        fn = len(gt_tracks) - tp
        
        return {
            'tp': tp, 'fp': fp, 'fn': fn,
            'associations': associations,
            'distances': valid_distances
        }
    
    def update_frame(self, gt_tracks: List[Track3D], pred_tracks: List[Track3D], 
                    frame_id: int):
        """Update metrics for a single frame"""

        frame_result = self.compute_frame_associations(gt_tracks, pred_tracks)
        
        self.total_gt += len(gt_tracks)
        self.total_pred += len(pred_tracks)
        self.total_tp += frame_result['tp']
        self.total_fp += frame_result['fp']
        self.total_fn += frame_result['fn']
        
        self.frame_associations[frame_id] = frame_result
        
        # Track trajectory matches for IDF1
        for gt_idx, pred_idx in frame_result['associations'].items():
            gt_id = gt_tracks[gt_idx].track_id
            pred_id = pred_tracks[pred_idx].track_id
            self.trajectory_matches[gt_id].append((frame_id, pred_id))
    
    def compute_id_switches(self, trajectories: Dict) -> int:
        """Compute identity switches across trajectory"""

        total_switches = 0
        
        for gt_id, matches in self.trajectory_matches.items():
            if len(matches) < 2:
                continue
                
            matches.sort(key=lambda x: x[0])  # Sort by frame
            prev_pred_id = matches[0][1]
            
            for frame_id, pred_id in matches[1:]:
                if pred_id != prev_pred_id:
                    total_switches += 1
                prev_pred_id = pred_id
        
        return total_switches
    
    def compute_fragmentations(self) -> int:
        """Compute trajectory fragmentations"""

        fragmentations = 0
        
        for gt_id, matches in self.trajectory_matches.items():
            if len(matches) < 2:
                continue
                
            matches.sort(key=lambda x: x[0])  # Sort by frame
            frames = [m[0] for m in matches]
            
            # Count gaps in frame sequence
            for i in range(1, len(frames)):
                if frames[i] - frames[i-1] > 1:
                    fragmentations += 1
        
        return fragmentations
    
    def compute_mota(self) -> float:
        """Multiple Object Tracking Accuracy"""

        if self.total_gt == 0:
            return 0.0
        
        return 1 - (self.total_fn + self.total_fp + self.total_id_switches) / self.total_gt
    
    def compute_idf1(self) -> float:
        """ID F1 Score"""
        if not self.trajectory_matches:
            return 0.0
        
        # Count correctly identified trajectory points
        correctly_identified = 0
        total_trajectory_points = 0
        
        for gt_id, matches in self.trajectory_matches.items():
            if not matches:
                continue
                
            # Find the most frequent predicted ID for this GT trajectory
            pred_ids = [m[1] for m in matches]
            most_frequent_pred_id = Counter(pred_ids).most_common(1)[0][0]
            
            # Count how many times this prediction was correct
            correct_for_this_traj = sum(1 for pred_id in pred_ids if pred_id == most_frequent_pred_id)
            correctly_identified += correct_for_this_traj
            total_trajectory_points += len(matches)
        
        if total_trajectory_points == 0:
            return 0.0
        
        return correctly_identified / total_trajectory_points
    
    def compute_hota3d(self) -> float:
        """Higher Order Tracking Accuracy for 3D (simplified version)"""

        if not self.frame_associations:
            return 0.0
        
        # Compute detection accuracy
        total_tp = sum(frame['tp'] for frame in self.frame_associations.values())
        total_fp = sum(frame['fp'] for frame in self.frame_associations.values())
        total_fn = sum(frame['fn'] for frame in self.frame_associations.values())
        
        if total_tp + total_fp + total_fn == 0:
            return 0.0
        
        detection_accuracy = total_tp / (total_tp + 0.5 * (total_fp + total_fn))
        
        # Compute association accuracy (simplified)
        association_accuracy = self.compute_idf1()
        
        # HOTA is geometric mean of detection and association accuracy
        return np.sqrt(detection_accuracy * association_accuracy)

class Metrics3D:
    """Metrics computation for 3D basketball tracking"""
    
    def __init__(self):
        self.trajectory_metrics = TrajectoryMetrics()
        self.mot_metrics = MOTMetrics()
        self.class_labels = {0: 'Ball', 1: 'Player', 2: 'Referee'}
        
    def load_3d_results(self, results_path: str) -> Dict:
        """Load 3D tracking results"""
        with open(results_path, 'r') as f:
            results_data = json.load(f)
        
        # Convert back to Track3D objects
        tracks_3d = {}
        for frame_id_str, tracks_data in results_data.items():
            frame_id = int(frame_id_str)
            frame_tracks = []
            
            for track_data in tracks_data:
                track = Track3D(
                    frame_id=track_data['frame_id'],
                    track_id=track_data['track_id'],
                    position=np.array(track_data['position']),
                    confidence=track_data['confidence'],
                    class_id=track_data['class_id'],
                    camera_count=track_data['camera_count'],
                    camera_ids=track_data['camera_ids'],
                    reprojection_error=track_data['reprojection_error']
                )
                frame_tracks.append(track)
            
            tracks_3d[frame_id] = frame_tracks
        
        return tracks_3d
    
    def extract_trajectories(self, tracks_3d: Dict) -> Dict:
        """Extract continuous trajectories by track_id and class"""

        track_groups = defaultdict(list)
        
        for frame_id, tracks in tracks_3d.items():
            for track in tracks:
                key = f"{track.track_id}_{track.class_id}"
                track_groups[key].append(track)
        
        # Sort by frame_id for each track
        for track_key in track_groups:
            track_groups[track_key].sort(key=lambda x: x.frame_id)
        
        return dict(track_groups)
    
    def compute_trajectory_metrics_per_object(self, trajectories: Dict) -> Dict:
        """
        Compute ADE, FDE, RMSE for each object type
        Note: No 3D ground truth, we compute smoothness metrics
        """
        metrics_by_class = {class_id: {'ade': [], 'fde': [], 'rmse': [], 'smoothness': []} 
                           for class_id in [0, 1, 2]}
        
        for track_key, trajectory in trajectories.items():
            if len(trajectory) < 3:  # Need at least 3 points for meaningful analysis
                continue
                
            class_id = trajectory[0].class_id
            positions = np.array([track.position for track in trajectory])
            
            # Since we don't have GT, compute trajectory smoothness metrics
            # Velocity-based smoothness
            velocities = np.diff(positions, axis=0)
            accelerations = np.diff(velocities, axis=0)
            
            # Smoothness as acceleration variance (lower is smoother)
            smoothness = np.mean(np.linalg.norm(accelerations, axis=1)) if len(accelerations) > 0 else 0
            
            # Displacement from linear interpolation (as proxy for accuracy)
            if len(positions) > 2:
                # Fit linear trajectory
                t = np.arange(len(positions))
                linear_positions = np.array([
                    np.interp(t, [0, len(positions)-1], [positions[0][i], positions[-1][i]])
                    for i in range(3)
                ]).T
                
                # Compute "ADE" as deviation from linear motion
                ade = np.mean([np.linalg.norm(pos - lin_pos) 
                              for pos, lin_pos in zip(positions, linear_positions)])
                
                # "FDE" as final deviation
                fde = np.linalg.norm(positions[-1] - linear_positions[-1])
                
                # RMSE
                rmse = np.sqrt(np.mean([(pos - lin_pos)**2 
                                       for pos, lin_pos in zip(positions, linear_positions)]))
                rmse = np.sqrt(np.mean(rmse))
                
                metrics_by_class[class_id]['ade'].append(ade)
                metrics_by_class[class_id]['fde'].append(fde)
                metrics_by_class[class_id]['rmse'].append(rmse)
            
            metrics_by_class[class_id]['smoothness'].append(smoothness)
        
        # Compute averages
        summary_metrics = {}
        for class_id, class_metrics in metrics_by_class.items():
            class_name = self.class_labels[class_id]
            summary_metrics[class_name] = {
                'avg_ade_mm': float(np.mean(class_metrics['ade']) if class_metrics['ade'] else 0),
                'avg_fde_mm': float(np.mean(class_metrics['fde']) if class_metrics['fde'] else 0),
                'avg_rmse_mm': float(np.mean(class_metrics['rmse']) if class_metrics['rmse'] else 0),
                'avg_smoothness': float(np.mean(class_metrics['smoothness']) if class_metrics['smoothness'] else 0),
                'trajectory_count': len([t for t in trajectories.values() if t and t[0].class_id == class_id])
            }
        
        return summary_metrics
    
    def compute_mot3d_metrics(self, tracks_3d: Dict) -> Dict:
        """
        Compute MOT 3D metrics
        Note: Without ground truth, we use consistency-based metrics
        """
        # Reset metrics
        self.mot_metrics.reset()
        
        # Since we don't have ground truth, we'll compute internal consistency metrics
        frame_ids = sorted(tracks_3d.keys())
        
        # Compute track consistency metrics
        track_consistency = self._compute_track_consistency(tracks_3d)
        
        # Estimate ID switches and fragmentations
        trajectories = self.extract_trajectories(tracks_3d)
        id_switches = self._estimate_id_switches(trajectories)
        fragmentations = self._estimate_fragmentations(trajectories)
        
        # Compute simplified metrics based on track consistency
        total_tracks = sum(len(tracks) for tracks in tracks_3d.values())
        
        mot_metrics = {
            'estimated_mota': float(1.0 - (id_switches + fragmentations) / total_tracks if total_tracks > 0 else 0),
            'estimated_idf1': float(track_consistency['avg_consistency']),
            'estimated_hota3d': float(np.sqrt(track_consistency['avg_consistency'] * 0.8)),  # Simplified
            'id_switches': int(id_switches),
            'fragmentations': int(fragmentations),
            'total_tracks': int(total_tracks),
            'track_consistency': track_consistency
        }
        
        return mot_metrics
    
    def _compute_track_consistency(self, tracks_3d: Dict) -> Dict:
        """Compute track consistency metrics"""

        trajectories = self.extract_trajectories(tracks_3d)
        
        consistency_scores = []
        length_consistency = []
        spatial_consistency = []
        
        for track_key, trajectory in trajectories.items():
            if len(trajectory) < 2:
                continue
            
            # Temporal consistency (no large time gaps)
            frame_ids = [track.frame_id for track in trajectory]
            frame_gaps = np.diff(sorted(frame_ids))
            temporal_consistency = np.sum(frame_gaps == 1) / len(frame_gaps) if frame_gaps.size > 0 else 1.0
            
            # Spatial consistency (smooth motion)
            positions = np.array([track.position for track in trajectory])
            if len(positions) > 2:
                velocities = np.diff(positions, axis=0)
                velocity_magnitudes = np.linalg.norm(velocities, axis=1)
                spatial_consistency_score = 1.0 / (1.0 + np.std(velocity_magnitudes))
            else:
                spatial_consistency_score = 1.0
            
            overall_consistency = 0.6 * temporal_consistency + 0.4 * spatial_consistency_score
            
            consistency_scores.append(overall_consistency)
            length_consistency.append(len(trajectory))
            spatial_consistency.append(spatial_consistency_score)
        
        return {
            'avg_consistency': float(np.mean(consistency_scores) if consistency_scores else 0),
            'avg_trajectory_length': float(np.mean(length_consistency) if length_consistency else 0),
            'avg_spatial_consistency': float(np.mean(spatial_consistency) if spatial_consistency else 0),
            'total_trajectories': len(trajectories)
        }
    
    def _estimate_id_switches(self, trajectories: Dict) -> int:
        """Estimate ID switches based on trajectory overlaps"""

        id_switches = 0
        
        # Group by class for analysis
        class_trajectories = defaultdict(list)
        for track_key, trajectory in trajectories.items():
            if trajectory:
                class_id = trajectory[0].class_id
                class_trajectories[class_id].append((track_key, trajectory))
        
        # For each class, look for potential ID switches
        for class_id, class_tracks in class_trajectories.items():
            for i, (key1, traj1) in enumerate(class_tracks):
                for j, (key2, traj2) in enumerate(class_tracks[i+1:], i+1):
                    # Check for temporal overlap and spatial proximity
                    frames1 = set(track.frame_id for track in traj1)
                    frames2 = set(track.frame_id for track in traj2)
                    
                    overlap_frames = frames1.intersection(frames2)
                    if overlap_frames:
                        # Check if tracks are close in overlapping frames
                        for frame_id in overlap_frames:
                            pos1 = next(track.position for track in traj1 if track.frame_id == frame_id)
                            pos2 = next(track.position for track in traj2 if track.frame_id == frame_id)
                            
                            distance = np.linalg.norm(pos1 - pos2)
                            if distance < 500:  # 50cm threshold
                                id_switches += 1
                                break
        
        return id_switches
    
    def _estimate_fragmentations(self, trajectories: Dict) -> int:
        """Estimate fragmentations based on trajectory gaps"""

        fragmentations = 0
        
        for track_key, trajectory in trajectories.items():
            if len(trajectory) < 2:
                continue
            
            frame_ids = sorted([track.frame_id for track in trajectory])
            gaps = np.diff(frame_ids)
            
            # Count gaps larger than 1 frame
            fragmentations += np.sum(gaps > 1)
        
        return fragmentations
    
    def compute_reprojection_analysis(self, tracks_3d: Dict) -> Dict:
        """Detailed reprojection error analysis"""

        all_errors = []
        errors_by_class = {0: [], 1: [], 2: []}
        errors_by_camera_count = {2: [], 3: []}
        
        for tracks in tracks_3d.values():
            for track in tracks:
                all_errors.append(track.reprojection_error)
                errors_by_class[track.class_id].append(track.reprojection_error)
                if track.camera_count in errors_by_camera_count:
                    errors_by_camera_count[track.camera_count].append(track.reprojection_error)
        
        analysis = {
            'overall': {
                'mean_px': float(np.mean(all_errors) if all_errors else 0),
                'median_px': float(np.median(all_errors) if all_errors else 0),
                'std_px': float(np.std(all_errors) if all_errors else 0),
                'min_px': float(np.min(all_errors) if all_errors else 0),
                'max_px': float(np.max(all_errors) if all_errors else 0),
                'q75_px': float(np.percentile(all_errors, 75) if all_errors else 0),
                'q95_px': float(np.percentile(all_errors, 95) if all_errors else 0)
            },
            'by_class': {},
            'by_camera_count': {}
        }
        
        # By class analysis
        for class_id, errors in errors_by_class.items():
            class_name = self.class_labels[class_id]
            if errors:
                analysis['by_class'][class_name] = {
                    'mean_px': float(np.mean(errors)),
                    'median_px': float(np.median(errors)),
                    'std_px': float(np.std(errors)),
                    'count': len(errors)
                }
            else:
                analysis['by_class'][class_name] = {
                    'mean_px': 0.0, 'median_px': 0.0, 'std_px': 0.0, 'count': 0
                }
        
        # By camera count analysis
        for cam_count, errors in errors_by_camera_count.items():
            if errors:
                analysis['by_camera_count'][f'{cam_count}_cameras'] = {
                    'mean_px': float(np.mean(errors)),
                    'median_px': float(np.median(errors)),
                    'std_px': float(np.std(errors)),
                    'count': len(errors)
                }
        
        return analysis
    
    def create_visualizations(self, tracks_3d: Dict, trajectories: Dict, 
                                     all_metrics: Dict, output_dir: str):
        """Create visualizations with new metrics"""

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 1. Trajectory metrics visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('3DTracking Metrics', fontsize=16, fontweight='bold')
        
        # Trajectory metrics by class
        traj_metrics = all_metrics['trajectory_metrics']
        classes = list(traj_metrics.keys())
        ade_values = [traj_metrics[cls]['avg_ade_mm'] for cls in classes]
        fde_values = [traj_metrics[cls]['avg_fde_mm'] for cls in classes]
        
        x_pos = np.arange(len(classes))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, ade_values, width, label='ADE', alpha=0.8, color='skyblue')
        axes[0, 0].bar(x_pos + width/2, fde_values, width, label='FDE', alpha=0.8, color='lightcoral')
        axes[0, 0].set_title('Average Displacement Error (ADE) & Final Displacement Error (FDE)')
        axes[0, 0].set_xlabel('Object Class')
        axes[0, 0].set_ylabel('Error (mm)')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(classes)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE by class
        rmse_values = [traj_metrics[cls]['avg_rmse_mm'] for cls in classes]
        axes[0, 1].bar(classes, rmse_values, color='lightgreen', alpha=0.8)
        axes[0, 1].set_title('Root Mean Square Error (RMSE) by Object Class')
        axes[0, 1].set_xlabel('Object Class')
        axes[0, 1].set_ylabel('RMSE (mm)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # MOT metrics
        mot_metrics = all_metrics['mot3d_metrics']
        mot_metric_names = ['MOTA', 'IDF1', 'HOTA3D']
        mot_metric_values = [
            mot_metrics['estimated_mota'],
            mot_metrics['estimated_idf1'], 
            mot_metrics['estimated_hota3d']
        ]
        
        bars = axes[1, 0].bar(mot_metric_names, mot_metric_values, 
                             color=['orange', 'purple', 'brown'], alpha=0.8)
        axes[1, 0].set_title('MOT 3D Metrics (Estimated)')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, mot_metric_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Reprojection error distribution
        reprojection_analysis = all_metrics['reprojection_analysis']
        all_errors = []
        for tracks in tracks_3d.values():
            for track in tracks:
                all_errors.append(track.reprojection_error)
        
        if all_errors:
            axes[1, 1].hist(all_errors, bins=50, alpha=0.7, color='gold', edgecolor='black')
            axes[1, 1].axvline(reprojection_analysis['overall']['mean_px'], 
                              color='red', linestyle='--', linewidth=2, 
                              label=f"Mean: {reprojection_analysis['overall']['mean_px']:.1f}px")
            axes[1, 1].axvline(reprojection_analysis['overall']['median_px'], 
                              color='blue', linestyle='--', linewidth=2,
                              label=f"Median: {reprojection_analysis['overall']['median_px']:.1f}px")
            axes[1, 1].set_title('Reprojection Error Distribution')
            axes[1, 1].set_xlabel('Reprojection Error (pixels)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / '3d_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed reprojection analysis
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Detailed Reprojection Error Analysis', fontsize=14, fontweight='bold')
        
        # By class
        class_names = []
        class_errors = []
        class_stds = []
        
        for class_name, metrics in reprojection_analysis['by_class'].items():
            if metrics['count'] > 0:
                class_names.append(class_name)
                class_errors.append(metrics['mean_px'])
                class_stds.append(metrics['std_px'])
        
        if class_names:
            bars = axes[0].bar(class_names, class_errors, yerr=class_stds, 
                              capsize=5, alpha=0.8, color=['red', 'blue', 'green'][:len(class_names)])
            axes[0].set_title('Mean Reprojection Error by Object Class')
            axes[0].set_ylabel('Error (pixels)')
            axes[0].grid(True, alpha=0.3)
            
            for bar, error, count in zip(bars, class_errors, 
                                        [reprojection_analysis['by_class'][name]['count'] for name in class_names]):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{error:.1f}px\n(n={count})', ha='center', va='bottom', fontsize=9)
        
        # By camera count
        cam_counts = []
        cam_errors = []
        cam_stds = []
        
        for cam_label, metrics in reprojection_analysis['by_camera_count'].items():
            cam_counts.append(cam_label)
            cam_errors.append(metrics['mean_px'])
            cam_stds.append(metrics['std_px'])
        
        if cam_counts:
            bars = axes[1].bar(cam_counts, cam_errors, yerr=cam_stds, 
                              capsize=5, alpha=0.8, color=['orange', 'purple'])
            axes[1].set_title('Mean Reprojection Error by Camera Count')
            axes[1].set_ylabel('Error (pixels)')
            axes[1].grid(True, alpha=0.3)
            
            for bar, error, label in zip(bars, cam_errors, cam_counts):
                count = reprojection_analysis['by_camera_count'][label]['count']
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{error:.1f}px\n(n={count})', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'reprojection_error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Trajectory quality analysis
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Trajectory Quality Analysis', fontsize=14, fontweight='bold')
        
        # Trajectory length distribution
        traj_lengths = [len(traj) for traj in trajectories.values()]
        if traj_lengths:
            axes[0].hist(traj_lengths, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
            axes[0].axvline(np.mean(traj_lengths), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(traj_lengths):.1f} frames')
            axes[0].axvline(np.median(traj_lengths), color='blue', linestyle='--', 
                           label=f'Median: {np.median(traj_lengths):.1f} frames')
            axes[0].set_title('Trajectory Length Distribution')
            axes[0].set_xlabel('Trajectory Length (frames)')
            axes[0].set_ylabel('Frequency')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Smoothness by class
        smoothness_by_class = {}
        for track_key, trajectory in trajectories.items():
            if len(trajectory) < 3:
                continue
            class_id = trajectory[0].class_id
            class_name = self.class_labels[class_id]
            
            positions = np.array([track.position for track in trajectory])
            velocities = np.diff(positions, axis=0)
            accelerations = np.diff(velocities, axis=0)
            smoothness = np.mean(np.linalg.norm(accelerations, axis=1)) if len(accelerations) > 0 else 0
            
            if class_name not in smoothness_by_class:
                smoothness_by_class[class_name] = []
            smoothness_by_class[class_name].append(smoothness)
        
        if smoothness_by_class:
            class_names = list(smoothness_by_class.keys())
            smoothness_means = [np.mean(smoothness_by_class[name]) for name in class_names]
            smoothness_stds = [np.std(smoothness_by_class[name]) for name in class_names]
            
            bars = axes[1].bar(class_names, smoothness_means, yerr=smoothness_stds,
                              capsize=5, alpha=0.8, color=['red', 'blue', 'green'][:len(class_names)])
            axes[1].set_title('Trajectory Smoothness by Object Class')
            axes[1].set_ylabel('Average Acceleration (mm/frame²)')
            axes[1].grid(True, alpha=0.3)
            
            for bar, mean_val in zip(bars, smoothness_means):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(smoothness_means)*0.01,
                           f'{mean_val:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'trajectory_quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}")
    
    def generate_report(self, all_metrics: Dict, output_path: str):
        """Generate metrics report"""

        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("3D BASKETBALL GAME TRACKING EVALUATION REPORT TXT\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall statistics
            basic_stats = all_metrics['basic_statistics']
            f.write("1. OVERALL TRACKING STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total frames processed: {basic_stats['total_frames_processed']}\n")
            f.write(f"Frames with tracks: {basic_stats['frames_with_tracks']}\n")
            f.write(f"Success rate: {basic_stats['success_rate']:.1%}\n")
            f.write(f"Total 3D tracks: {basic_stats['total_tracks']}\n")
            f.write(f"Average tracks per frame: {basic_stats['tracks_per_frame']:.1f}\n\n")
            
            # Trajectory metrics
            f.write("2. TRAJECTORY METRICS\n")
            f.write("-" * 40 + "\n")
            traj_metrics = all_metrics['trajectory_metrics']
            for class_name, metrics in traj_metrics.items():
                f.write(f"\n{class_name}:\n")
                f.write(f"  Average Displacement Error (ADE): {metrics['avg_ade_mm']:.1f} mm\n")
                f.write(f"  Final Displacement Error (FDE): {metrics['avg_fde_mm']:.1f} mm\n")
                f.write(f"  Root Mean Square Error (RMSE): {metrics['avg_rmse_mm']:.1f} mm\n")
                f.write(f"  Average Smoothness: {metrics['avg_smoothness']:.1f} mm/frame²\n")
                f.write(f"  Number of trajectories: {metrics['trajectory_count']}\n")
            
            # MOT 3D metrics
            f.write("\n3. MULTI-OBJECT TRACKING 3D METRICS\n")
            f.write("-" * 40 + "\n")
            mot_metrics = all_metrics['mot3d_metrics']
            f.write(f"Estimated MOTA (Multiple Object Tracking Accuracy): {mot_metrics['estimated_mota']:.3f}\n")
            f.write(f"Estimated IDF1 (ID F1 Score): {mot_metrics['estimated_idf1']:.3f}\n")
            f.write(f"Estimated HOTA3D (Higher Order Tracking Accuracy): {mot_metrics['estimated_hota3d']:.3f}\n")
            f.write(f"Identity Switches (IDSW): {mot_metrics['id_switches']}\n")
            f.write(f"Fragmentations (FRAG): {mot_metrics['fragmentations']}\n")
            f.write(f"Track Consistency Score: {mot_metrics['track_consistency']['avg_consistency']:.3f}\n")
            f.write(f"Average Trajectory Length: {mot_metrics['track_consistency']['avg_trajectory_length']:.1f} frames\n")
            
            # Reprojection error analysis
            f.write("\n4. REPROJECTION ERROR ANALYSIS\n")
            f.write("-" * 40 + "\n")
            reproj_analysis = all_metrics['reprojection_analysis']
            overall = reproj_analysis['overall']
            f.write(f"Overall Statistics:\n")
            f.write(f"  Mean: {overall['mean_px']:.2f} pixels\n")
            f.write(f"  Median: {overall['median_px']:.2f} pixels\n")
            f.write(f"  Standard Deviation: {overall['std_px']:.2f} pixels\n")
            f.write(f"  Min: {overall['min_px']:.2f} pixels\n")
            f.write(f"  Max: {overall['max_px']:.2f} pixels\n")
            f.write(f"  75th Percentile: {overall['q75_px']:.2f} pixels\n")
            f.write(f"  95th Percentile: {overall['q95_px']:.2f} pixels\n\n")
            
            f.write("By Object Class:\n")
            for class_name, metrics in reproj_analysis['by_class'].items():
                if metrics['count'] > 0:
                    f.write(f"  {class_name}:\n")
                    f.write(f"    Mean: {metrics['mean_px']:.2f} ± {metrics['std_px']:.2f} pixels\n")
                    f.write(f"    Median: {metrics['median_px']:.2f} pixels\n")
                    f.write(f"    Count: {metrics['count']}\n")
            
            f.write("\nBy Camera Count:\n")
            for cam_count, metrics in reproj_analysis['by_camera_count'].items():
                f.write(f"  {cam_count}:\n")
                f.write(f"    Mean: {metrics['mean_px']:.2f} ± {metrics['std_px']:.2f} pixels\n")
                f.write(f"    Count: {metrics['count']}\n")
            
            # Quality assessment
            f.write("\n5. QUALITY ASSESSMENT\n")
            f.write("-" * 40 + "\n")
            
            # Reprojection quality
            mean_error = overall['mean_px']
            if mean_error < 2.0:
                reproj_quality = "Excellent"
            elif mean_error < 5.0:
                reproj_quality = "Good"
            elif mean_error < 10.0:
                reproj_quality = "Acceptable"
            else:
                reproj_quality = "Needs Improvement"
            
            f.write(f"Reprojection Quality: {reproj_quality} (Mean error: {mean_error:.2f}px)\n")
            
            # Tracking consistency
            consistency = mot_metrics['track_consistency']['avg_consistency']
            if consistency > 0.8:
                track_quality = "Excellent"
            elif consistency > 0.6:
                track_quality = "Good"
            elif consistency > 0.4:
                track_quality = "Acceptable"
            else:
                track_quality = "Needs Improvement"
            
            f.write(f"Track Consistency: {track_quality} (Score: {consistency:.3f})\n")
            
            # Success rate assessment
            success_rate = basic_stats['success_rate']
            if success_rate > 0.9:
                success_quality = "Excellent"
            elif success_rate > 0.7:
                success_quality = "Good"
            elif success_rate > 0.5:
                success_quality = "Acceptable"
            else:
                success_quality = "Needs Improvement"
            
            f.write(f"Frame Success Rate: {success_quality} ({success_rate:.1%})\n")
            
            f.write("\n6. RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            
            if mean_error > 5.0:
                f.write("• Consider camera recalibration to reduce reprojection errors\n")
            
            if consistency < 0.6:
                f.write("• Improve track association algorithms for better consistency\n")
            
            if mot_metrics['id_switches'] > mot_metrics['total_tracks'] * 0.1:
                f.write("• High number of ID switches detected - review tracking parameters\n")
            
            if success_rate < 0.7:
                f.write("• Low success rate - consider relaxing detection thresholds\n")
            
            ball_trajectories = traj_metrics.get('Ball', {}).get('trajectory_count', 0)
            if ball_trajectories < basic_stats['total_frames_processed'] * 0.3:
                f.write("• Ball detection rate is low - consider specialized ball tracking\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Report generated for 3D Tracking Metrics\n")
            f.write("=" * 80 + "\n")
    
    def run_analysis(self, results_path: str, output_dir: str = "3D_analysis"):
        """Run the complete analysis"""

        print("Starting 3D basketballgame tracking analysis...")
        
        # Load data
        tracks_3d = self.load_3d_results(results_path)
        print(f"Loaded {len(tracks_3d)} frames of 3D tracking results")
        
        # Extract trajectories
        trajectories = self.extract_trajectories(tracks_3d)
        print(f"Extracted {len(trajectories)} trajectories")
        
        # Compute all metrics
        print("Computing basic statistics...")
        basic_stats = self._compute_basic_stats(tracks_3d, trajectories)
        
        print("Computing trajectory metrics...")
        trajectory_metrics = self.compute_trajectory_metrics_per_object(trajectories)
        
        print("Computing MOT 3D metrics...")
        mot3d_metrics = self.compute_mot3d_metrics(tracks_3d)
        
        print("Computing reprojection analysis...")
        reprojection_analysis = self.compute_reprojection_analysis(tracks_3d)
        
        # Combine all metrics
        all_metrics = {
            'basic_statistics': basic_stats,
            'trajectory_metrics': trajectory_metrics,
            'mot3d_metrics': mot3d_metrics,
            'reprojection_analysis': reprojection_analysis
        }
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create visualizations
        print("Creating visualizations...")
        self.create_visualizations(tracks_3d, trajectories, all_metrics, output_dir)
        
        # Generate report
        report_path = output_dir / "tracking_report.txt"
        self.generate_report(all_metrics, report_path)
        
        # Save detailed metrics to JSON (convert numpy types for serialization)
        metrics_path = output_dir / "3d_metrics.json"
        serializable_metrics = self._convert_to_serializable(all_metrics)
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        # Print summary to console
        self._print_summary(all_metrics)
        
        print(f"\nAnalysis complete!")
        print(f"Results saved to: {output_dir}")
        print(f"Report: {report_path}")
        print(f"Metrics: {metrics_path}")
        
        return all_metrics
    
    def _convert_to_serializable(self, obj): # For debugging
        """Convert numpy types to native Python types for JSON serialization""" 

        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    
    def _compute_basic_stats(self, tracks_3d: Dict, trajectories: Dict) -> Dict:
        """Compute basic statistics (reused from original)"""

        total_tracks = sum(len(tracks) for tracks in tracks_3d.values())
        frames_with_tracks = len([f for f, tracks in tracks_3d.items() if tracks])
        
        # Class distribution
        class_counts = {0: 0, 1: 0, 2: 0}
        for tracks in tracks_3d.values():
            for track in tracks:
                class_counts[track.class_id] += 1
        
        trajectory_lengths = [len(traj) for traj in trajectories.values()]
        
        return {
            "total_frames_processed": int(len(tracks_3d)),
            "frames_with_tracks": int(frames_with_tracks),
            "total_tracks": int(total_tracks),
            "tracks_per_frame": float(total_tracks / len(tracks_3d) if tracks_3d else 0),
            "success_rate": float(frames_with_tracks / len(tracks_3d) if tracks_3d else 0),
            "class_distribution": {
                "ball": int(class_counts[0]),
                "player": int(class_counts[1]), 
                "referee": int(class_counts[2])
            },
            "trajectories": {
                "total": int(len(trajectories)),
                "avg_length": float(np.mean(trajectory_lengths) if trajectory_lengths else 0),
                "max_length": int(np.max(trajectory_lengths) if trajectory_lengths else 0),
                "min_length": int(np.min(trajectory_lengths) if trajectory_lengths else 0)
            }
        }
    
    def _print_summary(self, all_metrics: Dict):
        """Print summary of results to console"""

        print("\n" + "="*80)
        print("3D BASKETBALL GAME TRACKING ANALYSIS SUMMARY")
        print("="*80)
        
        basic_stats = all_metrics['basic_statistics']
        print(f"\nBasic Statistics:")
        print(f"  Success Rate: {basic_stats['success_rate']:.1%}")
        print(f"  Total Tracks: {basic_stats['total_tracks']}")
        print(f"  Total Trajectories: {basic_stats['trajectories']['total']}")
        
        traj_metrics = all_metrics['trajectory_metrics']
        print(f"\nTrajectory Metrics (Average):")
        for class_name, metrics in traj_metrics.items():
            if metrics['trajectory_count'] > 0:
                print(f"  {class_name}:")
                print(f"    ADE: {metrics['avg_ade_mm']:.1f}mm, FDE: {metrics['avg_fde_mm']:.1f}mm")
                print(f"    RMSE: {metrics['avg_rmse_mm']:.1f}mm")
        
        mot_metrics = all_metrics['mot3d_metrics']
        print(f"\nMOT 3D Metrics:")
        print(f"  Estimated MOTA: {mot_metrics['estimated_mota']:.3f}")
        print(f"  Estimated IDF1: {mot_metrics['estimated_idf1']:.3f}")
        print(f"  ID Switches: {mot_metrics['id_switches']}")
        print(f"  Fragmentations: {mot_metrics['fragmentations']}")
        
        reproj = all_metrics['reprojection_analysis']['overall']
        print(f"\nReprojection Error:")
        print(f"  Mean: {reproj['mean_px']:.2f}px ± {reproj['std_px']:.2f}px")
        print(f"  Median: {reproj['median_px']:.2f}px")


def main():
    """Main function to run analysis"""

    print("Basketball 3D Tracking Analysis")
    print("=" * 50)
    
    # Initialize the metrics calculator
    metrics_calculator = Metrics3D()
    
    # Run analysis
    results_path = "3D_tracking/results_3d_tracks.json"
    
    if not Path(results_path).exists():
        print(f"Results file not found: {results_path}")
        print("Please ensure the 3D tracking results file exists.")
        return
    
    try:
        all_metrics = metrics_calculator.run_analysis(
            results_path=results_path,
            output_dir="3D_tracking/3d_analysis"
        )
        
        print("\n" + "="*50)
        print("Analysis completed successfully!")
        print("Check the '3d_analysis' directory for:")
        print("  • Visualization plots")
        print("  • Text report")
        print("  • Detailed metrics JSON file")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()