"""
3D Tracking Data Setup Module
Handles loading and preprocessing of 2D tracking data and camera parameters
for 3D triangulation.
"""

import json
import numpy as np
import cv2
import os
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CameraParams:
    """Camera parameters"""
    mtx: np.ndarray              # Camera intrinsic matrix
    dist: np.ndarray             # Distortion coefficients
    tvecs: np.ndarray            # Translation vector (world to camera)
    rvecs: np.ndarray            # Rotation vector (world to camera)
    camera_id: str               # Camera ID (2,4,13)


@dataclass
class Track2D:
    """2D tracking data"""
    frame_id: int
    track_id: int
    bbox: List[float]            # [x1, y1, x2, y2]
    confidence: float
    class_id: int                # 0=ball, 1=player, 2=referee
    center: Tuple[float, float]  # Center point of bounding box


@dataclass
class FrameData:
    """All tracking data in a single frame"""
    frame_id: int
    tracks_by_camera: Dict[str, List[Track2D]]


class DataLoader:
    """Loading and preprocessing of tracking data and camera parameters"""
    
    def __init__(self, base_path: str = "/Users/davide/Desktop/CV_Tracking"):
        self.base_path = Path(base_path)
        self.tracking_data_path = self.base_path / "2D_tracking" / "tracked"
        self.camera_data_path = self.base_path / "3D_tracking" / "camera_data"
        
        # Available cameras based on the data
        self.camera_ids = ["2", "4", "13"]
        self.camera_params = {}
        self.tracking_data = {}
        self.undistort_maps = {}
        
    def load_camera_parameters(self) -> Dict[str, CameraParams]:
        """Load camera calibration parameters for all cameras"""
        print("Loading camera parameters...")
        
        for cam_id in self.camera_ids:
            calib_file = self.camera_data_path / f"cam_{cam_id}" / "calib" / "camera_calib.json"
            
            if not calib_file.exists():
                print(f"Warning: Camera calibration file not found for camera {cam_id}")
                continue
                
            with open(calib_file, 'r') as f:
                calib_data = json.load(f)
            
            # Convert to numpy arrays (IMPORTSNT)
            mtx = np.array(calib_data["mtx"], dtype=np.float32)
            dist = np.array(calib_data["dist"], dtype=np.float32)
            tvecs = np.array(calib_data["tvecs"], dtype=np.float32).reshape(3, 1)
            rvecs = np.array(calib_data["rvecs"], dtype=np.float32).reshape(3, 1)
            
            self.camera_params[cam_id] = CameraParams(
                mtx=mtx,
                dist=dist,
                tvecs=tvecs,
                rvecs=rvecs,
                camera_id=cam_id
            )
            
            print(f"Loaded camera {cam_id}: intrinsics shape {mtx.shape}, position {tvecs.flatten()}")
        
        return self.camera_params
    
    def create_undistortion_maps(self, image_size: Tuple[int, int] = (3840, 2160)):
        print("Creating undistortion maps...")
        
        for cam_id, params in self.camera_params.items():
            map_x, map_y = cv2.initUndistortRectifyMap(
                params.mtx, 
                params.dist, 
                None, 
                params.mtx, 
                image_size, 
                cv2.CV_32FC1
            )
            self.undistort_maps[cam_id] = (map_x, map_y)
            print(f"Created undistortion map for camera {cam_id}")
    
    def load_tracking_data(self) -> Dict[str, Dict[int, List[Track2D]]]:
        """Load 2D tracking data from JSON files"""
        print("Loading 2D tracking data...")
        
        # Map filename to camera ID
        file_to_cam = {
            "out2_tracks.json": "2",
            "out4_tracks.json": "4", 
            "out13_tracks.json": "13"
        }
        
        for filename, cam_id in file_to_cam.items():
            filepath = self.tracking_data_path / filename
            
            if not filepath.exists():
                print(f"Warning: Tracking file not found: {filename}")
                continue
            
            with open(filepath, 'r') as f:
                raw_data = json.load(f)
            
            # Process tracking data
            camera_tracks = {}
            for frame_data in raw_data:
                frame_id = frame_data["frame"]
                tracks = []
                
                for track in frame_data["tracks"]:
                    bbox = track["bbox"]
                    center_x = (bbox[0] + bbox[2]) / 2.0
                    center_y = (bbox[1] + bbox[3]) / 2.0
                    
                    track_2d = Track2D(
                        frame_id=frame_id,
                        track_id=track["id"],
                        bbox=bbox,
                        confidence=track["confidence"],
                        class_id=track["class_id"],  # 0=ball, 1=player, 2=referee
                        center=(center_x, center_y)
                    )
                    tracks.append(track_2d)
                
                camera_tracks[frame_id] = tracks
            
            self.tracking_data[cam_id] = camera_tracks
            num_frames = len(camera_tracks)
            total_tracks = sum(len(tracks) for tracks in camera_tracks.values())
            print(f"Loaded camera {cam_id}: {num_frames} frames, {total_tracks} total tracks")
        
        return self.tracking_data
    
    def rectify_2d_points(self, points: np.ndarray, camera_id: str) -> np.ndarray:
        """
        Apply rectification transformation to 2D points using undistortPoints
        
        Args:
            points: Array of shape (N, 2) containing [x, y] coordinates
            camera_id: Camera identifier
            
        Returns:
            Rectified points of shape (N, 2)
        """
        if camera_id not in self.camera_params:
            raise ValueError(f"Camera parameters not available for camera {camera_id}")
        
        params = self.camera_params[camera_id]
        
        # Reshape points for cv2.undistortPoints (needs (N, 1, 2))
        points_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        
        # Apply undistortion
        rectified_points = cv2.undistortPoints(
            points_reshaped, 
            params.mtx, 
            params.dist, 
            None, 
            params.mtx
        )
        
        # Reshape back to (N, 2)
        return rectified_points.reshape(-1, 2)
    
    def get_synchronized_tracks(self, frame_id: int, track_id: int) -> Dict[str, Track2D]:
        """
        Get synchronized tracks across all cameras for a specific frame and track ID
        
        Args:
            frame_id: Frame number
            track_id: Track ID to synchronize
            
        Returns:
            Dictionary mapping camera_id to Track2D object
        """
        synchronized_tracks = {}
        
        for cam_id in self.camera_ids:
            if cam_id in self.tracking_data and frame_id in self.tracking_data[cam_id]:
                # Find track with matching ID
                for track in self.tracking_data[cam_id][frame_id]:
                    if track.track_id == track_id:
                        synchronized_tracks[cam_id] = track
                        break
        
        return synchronized_tracks
    
    def get_frame_data(self, frame_id: int) -> FrameData:
        """Get all tracking data for a specific frame"""
        tracks_by_camera = {}
        
        for cam_id in self.camera_ids:
            if cam_id in self.tracking_data and frame_id in self.tracking_data[cam_id]:
                tracks_by_camera[cam_id] = self.tracking_data[cam_id][frame_id]
            else:
                tracks_by_camera[cam_id] = []
        
        return FrameData(frame_id=frame_id, tracks_by_camera=tracks_by_camera)
    
    def get_frame_range(self) -> Tuple[int, int]:
        """Get the range of available frames across all cameras"""
        all_frames = set()
        for cam_data in self.tracking_data.values():
            all_frames.update(cam_data.keys())
        
        if not all_frames:
            return 0, 0
        
        return min(all_frames), max(all_frames)
    
    def get_all_available_frames(self) -> Set[int]:
        """Get all frame IDs that have data in at least one camera"""
        all_frames = set()
        for cam_data in self.tracking_data.values():
            all_frames.update(cam_data.keys())
        return all_frames
    
    def apply_rectification_to_tracks(self):
        """Apply rectification to all loaded tracking data"""
        print("Applying rectification to tracking data...")
        
        for cam_id in self.camera_ids:
            if cam_id not in self.tracking_data or cam_id not in self.camera_params:
                continue
            
            total_rectified = 0
            for frame_id, tracks in self.tracking_data[cam_id].items():
                for track in tracks:
                    # Rectify center point
                    original_point = np.array([[track.center[0], track.center[1]]], dtype=np.float32)
                    rectified_point = self.rectify_2d_points(original_point, cam_id)
                    track.center = (rectified_point[0, 0], rectified_point[0, 1])
                    
                    # Rectify bounding box corners
                    bbox_points = np.array([
                        [track.bbox[0], track.bbox[1]],  # top-left
                        [track.bbox[2], track.bbox[3]]   # bottom-right
                    ], dtype=np.float32)
                    
                    rectified_bbox = self.rectify_2d_points(bbox_points, cam_id)
                    track.bbox = [
                        rectified_bbox[0, 0], rectified_bbox[0, 1],
                        rectified_bbox[1, 0], rectified_bbox[1, 1]
                    ]
                    total_rectified += 1
            
            print(f"Rectified {total_rectified} tracks for camera {cam_id}")


def main():
    """Example usage of the DataLoader class"""
    loader = DataLoader()
    
    camera_params = loader.load_camera_parameters()
    print(f"Loaded {len(camera_params)} cameras")
    
    loader.create_undistortion_maps()
    
    tracking_data = loader.load_tracking_data()
    print(f"Loaded tracking data for {len(tracking_data)} cameras")
    
    loader.apply_rectification_to_tracks()
    
    start_frame, end_frame = loader.get_frame_range()
    all_frames = loader.get_all_available_frames()
    print(f"Frame range: {start_frame} to {end_frame}")
    print(f"Total frames available: {len(all_frames)}")
    
    for cam_id in loader.camera_ids:
        if cam_id in loader.tracking_data:
            cam_frames = len(loader.tracking_data[cam_id])
            print(f"Camera {cam_id}: {cam_frames} frames")
    
    return loader


if __name__ == "__main__":
    main()
