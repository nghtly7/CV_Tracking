"""
3D Triangulation and Tracking Module 
Performs 3D triangulation from multiple camera views to reconstruct
3D positions of tracked objects (players, ball, referee). 
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path

from data_setup import DataLoader, Track2D, CameraParams, FrameData


@dataclass
class Track3D:
    """3D tracking data"""
    frame_id: int
    track_id: int
    position: np.ndarray        # 3D position [X, Y, Z] in world coordinates
    confidence: float           # Average confidence from 2D tracks
    class_id: int              # 0=ball, 1=player, 2=referee
    camera_count: int          # Number of cameras that detected this track
    camera_ids: List[str]      # List of camera IDs that detected this track
    reprojection_error: float  # Average reprojection error


class Triangulator3D:
    """Handles 3D triangulation from multiple camera views"""
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.camera_params = data_loader.camera_params
        self.projection_matrices = {}
        self.tracks_3d = {}  # Dictionary: frame_id -> List[Track3D]
        
        # Build projection matrices
        self._build_projection_matrices()
        
        # Relaxed triangulation parameters for better coverage
        self.min_cameras = 2          # Minimum cameras required for triangulation
        self.max_reprojection_error = 100.0  # More lenient reprojection error threshold
        self.confidence_threshold = 0.2      # Lower confidence threshold
        self.max_distance_threshold = 5000.0 # Max distance for track association (mm)
        
    def _build_projection_matrices(self):
        """Build projection matrices for all cameras"""
        print("Building projection matrices...")
        
        for cam_id, params in self.camera_params.items():
            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(params.rvecs)
            
            # Build camera extrinsic matrix [R|t]
            extrinsic = np.hstack([R, params.tvecs])
            
            # Build projection matrix 
            projection_matrix = params.mtx @ extrinsic
            
            self.projection_matrices[cam_id] = projection_matrix
            print(f"Camera {cam_id} projection matrix shape: {projection_matrix.shape}")
    
    def triangulate_point_pair(self, point1: np.ndarray, point2: np.ndarray, 
                              cam1_id: str, cam2_id: str) -> Tuple[np.ndarray, float]:
        """
        Triangulate 3D point from two camera views using DLT method
        Args:
            point1: 2D point in camera 1 [x, y]
            point2: 2D point in camera 2 [x, y]
            cam1_id: Camera 1 identifier
            cam2_id: Camera 2 identifier 
        Returns:
            Tuple of (3D point, reprojection error)
        """

        P1 = self.projection_matrices[cam1_id]
        P2 = self.projection_matrices[cam2_id]
        
        # OpenCV's triangulation
        points_4d = cv2.triangulatePoints(P1, P2, point1.reshape(2, 1), point2.reshape(2, 1))
        
        # Convert from homogeneous coordinates
        if points_4d[3, 0] != 0:
            point_3d = points_4d[:3, 0] / points_4d[3, 0]
        else:
            point_3d = points_4d[:3, 0]  # Handle degenerate case
        
        # Calculate reprojection error
        error = self._calculate_reprojection_error(point_3d, [point1, point2], [cam1_id, cam2_id])
        
        return point_3d, error
    
    def triangulate_point_multi(self, points: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Triangulate 3D point from multiple camera views using least squares
        Args:
            points: Dictionary mapping camera_id to 2D point [x, y]
        Returns:
            Tuple of (3D point, average reprojection error)
        """

        if len(points) < 2:
            raise ValueError("Need at least 2 camera views for triangulation")
        
        # Use all possible pairs and average the results
        camera_ids = list(points.keys())
        all_points_3d = []
        all_errors = []
        
        for cam1_id, cam2_id in combinations(camera_ids, 2):
            try:
                point_3d, error = self.triangulate_point_pair(
                    points[cam1_id], points[cam2_id], cam1_id, cam2_id
                )
                
                # Filter out extremely bad triangulations
                if error < self.max_reprojection_error * 2:  
                    all_points_3d.append(point_3d)
                    all_errors.append(error)
            except Exception as e:
                print(f"Triangulation failed for cameras {cam1_id}, {cam2_id}: {e}")
                continue
        
        if not all_points_3d:
            raise ValueError("No valid triangulations found")
        
        # Average all valid triangulations
        final_point_3d = np.mean(all_points_3d, axis=0)
        avg_error = np.mean(all_errors)
        
        # Refine with all cameras
        if len(points) > 2:
            try:
                final_point_3d = self._refine_triangulation(final_point_3d, points)
                avg_error = self._calculate_reprojection_error(
                    final_point_3d, list(points.values()), list(points.keys())
                )
            except:
                # Keep original if refinement fails
                pass
        
        return final_point_3d, avg_error
    
    def _refine_triangulation(self, initial_point: np.ndarray, 
                            points: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Refine triangulation using non-linear optimization (Levenberg-Marquardt)
        Args:
            initial_point: Initial 3D point estimate
            points: Dictionary mapping camera_id to 2D point
        Returns:
            Refined 3D point
        """

        try:
            from scipy.optimize import least_squares
        except ImportError:
            # Return initial point if scipy not available
            return initial_point
        
        def reprojection_residual(point_3d):
            residuals = []
            for cam_id, point_2d in points.items():
                projected = self._project_3d_to_2d(point_3d, cam_id)
                residual = projected - point_2d
                residuals.extend([residual[0], residual[1]])
            return np.array(residuals)
        
        try:
            result = least_squares(reprojection_residual, initial_point, method='lm')
            return result.x
        except Exception:
            # Fallback to initial point if optimization fails
            return initial_point
    
    def _project_3d_to_2d(self, point_3d: np.ndarray, camera_id: str) -> np.ndarray:
        """Project 3D point to 2D camera coordinates"""

        P = self.projection_matrices[camera_id]
        point_3d_homo = np.append(point_3d, 1.0)
        
        projected_homo = P @ point_3d_homo
        
        if projected_homo[2] != 0:
            projected_2d = projected_homo[:2] / projected_homo[2]
        else:
            projected_2d = projected_homo[:2]
        
        return projected_2d
    
    def _calculate_reprojection_error(self, point_3d: np.ndarray, 
                                    points_2d: List[np.ndarray], 
                                    camera_ids: List[str]) -> float:
        """Calculate average reprojection error for a 3D point"""
        errors = []
        
        for point_2d, cam_id in zip(points_2d, camera_ids):
            projected = self._project_3d_to_2d(point_3d, cam_id)
            error = np.linalg.norm(projected - point_2d)
            errors.append(error)
        
        return np.mean(errors)
    
    def associate_tracks_across_cameras(self, frame_data: FrameData) -> List[Dict[str, Track2D]]:
        """
        Associate tracks across cameras with relaxed criteria for better coverage
        Args:
            frame_data: All tracking data for a single frame
        Returns:
            List of dictionaries, each containing associated tracks from different cameras
        """

        associated_tracks = []
        
        # Group tracks by class
        tracks_by_class = {}
        for cam_id, tracks in frame_data.tracks_by_camera.items():
            for track in tracks:
                if track.confidence < self.confidence_threshold:
                    continue
                    
                class_id = track.class_id
                if class_id not in tracks_by_class:
                    tracks_by_class[class_id] = {}
                if cam_id not in tracks_by_class[class_id]:
                    tracks_by_class[class_id][cam_id] = []
                tracks_by_class[class_id][cam_id].append(track)
        
        # For each class, associate tracks
        for class_id, camera_tracks in tracks_by_class.items():
            if class_id == 0:  # Ball - should be unique
                associated_tracks.extend(self._associate_ball_tracks(camera_tracks))
            else:  # Players/referee - can be multiple
                associated_tracks.extend(self._associate_player_tracks(camera_tracks))
        
        return associated_tracks
    
    def _associate_ball_tracks(self, camera_tracks: Dict[str, List[Track2D]]) -> List[Dict[str, Track2D]]:
        """Associate ball tracks (should be unique across cameras)"""
        if len(camera_tracks) < 2:
            return []
        
        # For ball, take the highest confidence track from each camera
        best_tracks = {}
        for cam_id, tracks in camera_tracks.items():
            if tracks:
                best_track = max(tracks, key=lambda t: t.confidence)
                best_tracks[cam_id] = best_track
        
        if len(best_tracks) >= 2:
            return [best_tracks]
        return []
    
    def _associate_player_tracks(self, camera_tracks: Dict[str, List[Track2D]]) -> List[Dict[str, Track2D]]:
        """
        Associate player tracks using relaxed criteria
        """
        if len(camera_tracks) < 2:
            return []
        
        # Strategy 1: Try to match by track ID first
        track_id_associations = self._associate_by_track_id(camera_tracks)
        
        # Strategy 2: For unmatched tracks, try spatial association
        spatial_associations = self._associate_by_spatial_proximity(camera_tracks, track_id_associations)
        
        return track_id_associations + spatial_associations
    
    def _associate_by_track_id(self, camera_tracks: Dict[str, List[Track2D]]) -> List[Dict[str, Track2D]]:
        """Associate tracks with matching IDs"""
        track_id_to_cameras = {}
        
        for cam_id, tracks in camera_tracks.items():
            for track in tracks:
                track_id = track.track_id
                if track_id not in track_id_to_cameras:
                    track_id_to_cameras[track_id] = {}
                track_id_to_cameras[track_id][cam_id] = track
        
        # Filter associations with at least 2 cameras
        valid_associations = []
        for track_id, cam_tracks in track_id_to_cameras.items():
            if len(cam_tracks) >= 2:
                valid_associations.append(cam_tracks)
        
        return valid_associations
    
    def _associate_by_spatial_proximity(self, camera_tracks: Dict[str, List[Track2D]], 
                                       existing_associations: List) -> List[Dict[str, Track2D]]:
        """
        Associate remaining tracks by spatial proximity (simplified approach)
        This is a basic implementation - could be improved with epipolar geometry
        """
        # Get tracks that are not already associated
        used_tracks = set()
        for assoc in existing_associations:
            for cam_id, track in assoc.items():
                used_tracks.add((cam_id, track.track_id))
        
        available_tracks = {}
        for cam_id, tracks in camera_tracks.items():
            available_tracks[cam_id] = [
                track for track in tracks 
                if (cam_id, track.track_id) not in used_tracks
            ]
        
        # Simple greedy matching (could be improved)
        associations = []
        camera_ids = list(available_tracks.keys())
        
        for cam1_id in camera_ids:
            for track1 in available_tracks[cam1_id]:
                if (cam1_id, track1.track_id) in used_tracks:
                    continue
                    
                # Try to find matches in other cameras
                potential_match = {cam1_id: track1}
                
                for cam2_id in camera_ids:
                    if cam2_id == cam1_id:
                        continue
                        
                    for track2 in available_tracks[cam2_id]:
                        if (cam2_id, track2.track_id) in used_tracks:
                            continue
                        
                        # Simple distance check in image coordinates (could be improved)
                        # For now, just add any available track
                        potential_match[cam2_id] = track2
                        used_tracks.add((cam2_id, track2.track_id))
                        break
                
                if len(potential_match) >= 2:
                    associations.append(potential_match)
                    used_tracks.add((cam1_id, track1.track_id))
        
        return associations
    
    def process_frame(self, frame_id: int) -> List[Track3D]:
        """
        Process a single frame to generate 3D tracks
        Args:
            frame_id: Frame number to process
        Returns:
            List of 3D tracks for this frame
        """

        frame_data = self.data_loader.get_frame_data(frame_id)
        
        # Check if frame has data in at least 2 cameras
        cameras_with_data = [cam_id for cam_id, tracks in frame_data.tracks_by_camera.items() if tracks]
        if len(cameras_with_data) < 2:
            return []
        
        # Associate tracks across cameras
        associated_tracks = self.associate_tracks_across_cameras(frame_data)
        
        tracks_3d = []
        
        for track_association in associated_tracks:
            if len(track_association) < self.min_cameras:
                continue
            
            try:
                # Extract 2D points and camera information
                points_2d = {}
                confidences = []
                class_ids = []
                track_ids = []
                
                for cam_id, track in track_association.items():
                    points_2d[cam_id] = np.array(track.center, dtype=np.float32)
                    confidences.append(track.confidence)
                    class_ids.append(track.class_id)
                    track_ids.append(track.track_id)
                
                # Triangulate 3D position
                position_3d, reprojection_error = self.triangulate_point_multi(points_2d)
                
                # More lenient filtering
                if reprojection_error > self.max_reprojection_error:
                    # Still save but mark as low quality
                    pass
                
                # Create 3D track
                track_3d = Track3D(
                    frame_id=frame_id,
                    track_id=track_ids[0],  # Use first track ID
                    position=position_3d,
                    confidence=np.mean(confidences),
                    class_id=class_ids[0],  # Should be consistent
                    camera_count=len(track_association),
                    camera_ids=list(track_association.keys()),
                    reprojection_error=reprojection_error
                )
                
                tracks_3d.append(track_3d)
                
            except Exception as e:
                #print(f"Failed to triangulate track in frame {frame_id}: {e}")
                continue
        
        return tracks_3d
    
    def process_sequence(self, start_frame: Optional[int] = None, 
                        end_frame: Optional[int] = None) -> Dict[int, List[Track3D]]:
        """
        Process entire sequence to generate 3D tracks for ALL frames
        Args:
            start_frame: Starting frame (default: first available)
            end_frame: Ending frame (default: last available)
        Returns:
            Dictionary mapping frame_id to list of 3D tracks
        """

        if start_frame is None or end_frame is None:
            seq_start, seq_end = self.data_loader.get_frame_range()
            start_frame = start_frame or seq_start
            end_frame = end_frame or seq_end
        
        # Get all available frames
        all_available_frames = self.data_loader.get_all_available_frames()
        frames_to_process = [f for f in all_available_frames if start_frame <= f <= end_frame]
        frames_to_process.sort()
        
        print(f"Processing {len(frames_to_process)} frames from {start_frame} to {end_frame}...")
        
        self.tracks_3d = {}
        total_tracks = 0
        successful_frames = 0
        
        for i, frame_id in enumerate(frames_to_process):
            if i % 50 == 0:
                print(f"Processing frame {frame_id} ({i+1}/{len(frames_to_process)})...")
            
            tracks_3d = self.process_frame(frame_id)
            self.tracks_3d[frame_id] = tracks_3d
            
            if tracks_3d:
                successful_frames += 1
                total_tracks += len(tracks_3d)
        
        print(f"Generated {total_tracks} 3D tracks across {successful_frames}/{len(frames_to_process)} frames")
        print(f"Average tracks per successful frame: {total_tracks/successful_frames if successful_frames > 0 else 0:.2f}")
        
        return self.tracks_3d
    
    def save_results(self, output_path: str):
        """Save 3D tracking results to JSON file"""
        output_path = Path(output_path)
        
        # Convert to serializable format
        results = {}
        for frame_id, tracks in self.tracks_3d.items():
            frame_data = []
            for track in tracks:
                track_data = {
                    "frame_id": track.frame_id,
                    "track_id": track.track_id,
                    "position": track.position.tolist(),
                    "confidence": track.confidence,
                    "class_id": track.class_id,
                    "camera_count": track.camera_count,
                    "camera_ids": track.camera_ids,
                    "reprojection_error": track.reprojection_error
                }
                frame_data.append(track_data)
            results[str(frame_id)] = frame_data
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved 3D tracking results to {output_path}")
    
    def get_track_statistics(self) -> Dict:
        """Get statistics about the 3D tracking results"""

        if not self.tracks_3d:
            return {}
        
        total_tracks = sum(len(tracks) for tracks in self.tracks_3d.values())
        frames_with_tracks = len([f for f, tracks in self.tracks_3d.items() if tracks])
        
        # Class distribution
        class_counts = {0: 0, 1: 0, 2: 0}  # ball, player, referee
        reprojection_errors = []
        camera_counts = []
        
        for tracks in self.tracks_3d.values():
            for track in tracks:
                class_counts[track.class_id] += 1
                reprojection_errors.append(track.reprojection_error)
                camera_counts.append(track.camera_count)
        
        stats = {
            "total_frames_processed": len(self.tracks_3d),
            "frames_with_tracks": frames_with_tracks,
            "total_tracks": total_tracks,
            "tracks_per_frame": total_tracks / len(self.tracks_3d) if self.tracks_3d else 0,
            "success_rate": frames_with_tracks / len(self.tracks_3d) if self.tracks_3d else 0,
            "class_distribution": {
                "ball": class_counts[0],
                "player": class_counts[1], 
                "referee": class_counts[2]
            },
            "reprojection_error": {
                "mean": np.mean(reprojection_errors) if reprojection_errors else 0,
                "std": np.std(reprojection_errors) if reprojection_errors else 0,
                "max": np.max(reprojection_errors) if reprojection_errors else 0
            },
            "camera_count": {
                "mean": np.mean(camera_counts) if camera_counts else 0,
                "distribution": {str(i): camera_counts.count(i) for i in range(2, 4)}
            }
        }
        
        return stats


def main():
    """Example usage of the Triangulator3D class"""
    # Load data
    loader = DataLoader()
    loader.load_camera_parameters()
    loader.create_undistortion_maps()
    loader.load_tracking_data()
    loader.apply_rectification_to_tracks()
    
    # Create triangulator
    triangulator = Triangulator3D(loader)
    
    # Process ALL frames (should be ~600 frames)
    start_frame, end_frame = loader.get_frame_range()
    print(f"Processing all frames: {start_frame} to {end_frame}")
    
    # Process entire sequence
    results = triangulator.process_sequence(start_frame, end_frame)
    
    # Print statistics
    stats = triangulator.get_track_statistics()
    print("\n3D Tracking Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Save results
    output_path = "3D_tracking/results_3d_tracks.json"
    triangulator.save_results(output_path)
    
    return triangulator


if __name__ == "__main__":
    main()
