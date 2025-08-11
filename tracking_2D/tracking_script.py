"""
Basketball Tracking Script using YOLOv8 + DeepSORT
Performs object detection and tracking on basketball videos at native 25fps.
Uses fine-tuned YOLOv8 weights for detection and DeepSORT for tracking.


"""

import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort


def setup_directories(video_name):
    """Create necessary output directories for a specific video."""
    results_dir = Path("results")
    video_results_dir = results_dir / video_name
    frames_dir = video_results_dir / "frames"
    labels_dir = video_results_dir / "labels"
    
    results_dir.mkdir(exist_ok=True)
    video_results_dir.mkdir(exist_ok=True)
    frames_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    return video_results_dir, frames_dir, labels_dir

def load_model(weights_path):
    """Load the fine-tuned YOLOv8 model."""
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}")
        print("Please ensure you have trained the model first.")
        return None
    
    print(f"Loading model from {weights_path}...")
    model = YOLO(weights_path)
    print("Model loaded successfully!")
    return model

def initialize_tracker():
    """Initialize DeepSORT tracker with optimized parameters for basketball."""
    print("Initializing DeepSORT tracker...")
    
    # DeepSORT configuration optimized for basketball tracking
    tracker = DeepSort(
        max_age=15,              # Reduced from 30 - shorter track lifetime
        n_init=2,                # Reduced from 3 - faster track confirmation
        nms_max_overlap=0.7,     # Reduced from 1.0 - better overlap handling
        max_cosine_distance=0.3, # Reduced from 0.4 - stricter appearance matching
        nn_budget=100,           # Limited budget for appearance descriptors
        override_track_class=None,
        embedder="mobilenet",    # Feature embedder
        half=True,               # Use FP16
        bgr=True,                # Input format is BGR
        embedder_gpu=torch.cuda.is_available(),  # Use GPU for embedder if available
        embedder_model_name=None,
        embedder_wts=None,
        polygon=False,
        today=None
    )
    
    print("DeepSORT tracker initialized with optimized parameters!")
    return tracker

def process_detections(results, confidence_threshold=0.5, frame_count=0):
    """
    Process YOLO detection results for DeepSORT input.
    Returns list of detections in format: [[x1, y1, x2, y2, confidence, class_id], ...]
    """
    detections = []
    
    try:
        # Handle the case where results is a list
        if isinstance(results, list):
            result = results[0]
        else:
            result = results
        
        # Check if there are any detections
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            
            # Check if boxes tensor is not empty
            if len(boxes) > 0:
                # Get all data at once and convert to numpy
                xyxy = boxes.xyxy.cpu().numpy()  # Box coordinates
                conf = boxes.conf.cpu().numpy()  # Confidence scores
                cls = boxes.cls.cpu().numpy()    # Class IDs
                
                # Ensure we have the right dimensions
                if xyxy.ndim == 2 and xyxy.shape[1] == 4:
                    # Process each detection
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i]
                        confidence = float(conf[i])
                        class_id = int(cls[i])
                        
                        # Filter by confidence
                        if confidence >= confidence_threshold:
                            detections.append([float(x1), float(y1), float(x2), float(y2), confidence, class_id])
        
    except Exception as e:
        print(f"Error in process_detections: {e}")
    
    return detections

def draw_tracks(frame, tracks, class_names):
    """
    Draw tracking results on frame.
    """
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        ltrb = track.to_ltrb()
        
        # Get class ID from track
        try:
            class_id = track.get_det_class()
            if class_id is None:
                class_id = 1  # Default to Player
        except:
            # Fallback if get_det_class doesn't work
            class_id = getattr(track, 'det_class', 1)
        
        x1, y1, x2, y2 = map(int, ltrb)
        
        # Validate bounding box
        frame_height, frame_width = frame.shape[:2]
        width = x2 - x1
        height = y2 - y1
        
        # Skip unreasonably large or invalid boxes
        if (width > frame_width * 0.5 or height > frame_height * 0.5 or 
            x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1):
            continue
        
        # Clamp coordinates to frame boundaries
        x1 = max(0, min(x1, frame_width - 1))
        x2 = max(x1 + 1, min(x2, frame_width))
        y1 = max(0, min(y1, frame_height - 1))
        y2 = max(y1 + 1, min(y2, frame_height))
        
        # Define colors for each class
        colors = {
            0: (0, 255, 0),    # Ball - Green
            1: (255, 0, 0),    # Player - Blue
            2: (0, 0, 255)     # Referee - Red
        }
        
        color = colors.get(class_id, (255, 255, 255))  # Default white
        class_name = class_names.get(class_id, f"Class_{class_id}")
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw track ID and class
        label = f"ID:{track_id} {class_name}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Background for text
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Text
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

def save_yolo_labels(tracks, frame_width, frame_height, labels_dir, frame_count):
    """
    Save tracking results in YOLOv8 format with track IDs.
    Format: class_id center_x center_y width height track_id (all normalized 0-1)
    """
    label_filename = labels_dir / f"frame_{frame_count:06d}.txt"
    
    with open(label_filename, 'w') as f:
        for track in tracks:
            # Save labels for both confirmed AND tentative tracks
            # This ensures we don't miss labels in early frames
            # DeepSORT track states: 1=Tentative, 2=Confirmed, 3=Deleted
            if hasattr(track, 'state') and track.state == 3:
                # Skip deleted tracks
                continue
                
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = ltrb
            
            # Get track ID
            track_id = track.track_id
            
            # Get class ID from track
            try:
                class_id = track.get_det_class()
                if class_id is None:
                    class_id = 1  # Default to Player
            except:
                # Fallback if get_det_class doesn't work
                class_id = getattr(track, 'det_class', 1)
            
            # Validate bounding box
            width = x2 - x1
            height = y2 - y1
            
            # Skip invalid boxes
            if width <= 0 or height <= 0 or x1 < 0 or y1 < 0:
                continue
                
            # Clamp coordinates to frame boundaries - FIXED VERSION
            x1 = max(0, min(x1, frame_width - 1))
            x2 = max(x1 + 1, min(x2, frame_width))  # Ensure x2 > x1
            y1 = max(0, min(y1, frame_height - 1))
            y2 = max(y1 + 1, min(y2, frame_height))  # Ensure y2 > y1
            
            # Recalculate width/height after clamping
            width = x2 - x1
            height = y2 - y1
            
            # Skip if box becomes invalid after clamping
            if width <= 0 or height <= 0:
                continue
            
            # Convert to YOLOv8 format (normalized center coordinates and dimensions)
            center_x = (x1 + x2) / 2.0 / frame_width
            center_y = (y1 + y2) / 2.0 / frame_height
            norm_width = width / frame_width
            norm_height = height / frame_height
            
            # Ensure values are within [0, 1] range
            center_x = max(0.0, min(1.0, center_x))
            center_y = max(0.0, min(1.0, center_y))
            norm_width = max(0.0, min(1.0, norm_width))
            norm_height = max(0.0, min(1.0, norm_height))
            
            # Write to file in YOLOv8 format with track ID
            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f} {track_id}\n")

def process_video(video_path, video_name, model, class_names, confidence_threshold=0.5, target_fps=25):
    """Process a single video with tracking."""
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"{'='*60}")
    
    # Setup directories for this video
    video_results_dir, frames_dir, labels_dir = setup_directories(video_name)
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return False
    
    # Initialize tracker (fresh for each video)
    tracker = initialize_tracker()
    
    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f"- Resolution: {width}x{height}")
    print(f"- FPS: {fps}")
    print(f"- Total frames: {total_frames}")
    print(f"- Duration: {total_frames/fps:.2f} seconds")
    
    # Setup video writer
    output_video_path = video_results_dir / "tracked_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, target_fps, (width, height))
    
    frame_count = 0
    print(f"\nStarting tracking for {video_name}...")
    print(f"Output video: {output_video_path}")
    print(f"Output frames: {frames_dir}")
    print(f"Output labels: {labels_dir}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            try:
                # Run YOLO detection
                results = model(frame, conf=confidence_threshold, verbose=False)
                
                # Process detections
                detections = process_detections(results, confidence_threshold, frame_count)
                
                # Convert detections to DeepSORT format (XYWH)
                deepsort_detections = []
                for det in detections:
                    x1, y1, x2, y2, conf, class_id = det
                    x, y, width, height = x1, y1, x2 - x1, y2 - y1
                    deepsort_detections.append([[x, y, width, height], conf, class_id])
                
                # Update tracker
                if len(deepsort_detections) > 0:
                    tracks = tracker.update_tracks(deepsort_detections, frame=frame)
                else:
                    tracks = tracker.update_tracks([], frame=frame)
                
                # Draw tracking results
                tracked_frame = draw_tracks(frame.copy(), tracks, class_names)
                
                # Add frame info
                confirmed_tracks = [t for t in tracks if t.is_confirmed()]
                info_text = f"Frame: {frame_count}/{total_frames} | Objects: {len(confirmed_tracks)}"
                cv2.putText(tracked_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save frame
                frame_filename = frames_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_filename), tracked_frame)
                
                # Save YOLOv8 format labels
                save_yolo_labels(tracks, width, height, labels_dir, frame_count)
                
                # Write to output video
                out.write(tracked_frame)
                
                # Progress update - clean single line output
                progress = (frame_count / total_frames) * 100
                print(f"\r[{video_name}] Processing frame {frame_count}/{total_frames} ({progress:.1f}%) - Detections: {len(detections)}, Tracks: {len(confirmed_tracks)}", end='', flush=True)
                    
            except Exception as frame_error:
                print(f"\nError processing frame {frame_count}: {frame_error}")
                import traceback
                traceback.print_exc()
                break
    
    except KeyboardInterrupt:
        print(f"\nTracking interrupted by user for {video_name}.")
        return False
    
    except Exception as e:
        print(f"\nError during tracking {video_name}: {str(e)}")
        return False
    
    finally:
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\n{video_name} tracking completed!")
        print(f"- Processed: {frame_count} frames")
        print(f"- Output video: {output_video_path}")
        print(f"- Individual frames: {frames_dir}")
        print(f"- YOLOv8 labels: {labels_dir}")
        print(f"- Total output frames: {len(list(frames_dir.glob('*.jpg')))}")
        print(f"- Total label files: {len(list(labels_dir.glob('*.txt')))}")
        
    return True

def main():
    """Main function to process all videos."""
    print("Basketball Tracking with YOLOv8 + DeepSORT")
    print("Processing Multiple Videos Automatically")
    print("=" * 60)
    
    # Configuration
    weights_path = "best.pt"
    confidence_threshold = 0.5
    target_fps = 25
    
    # Videos to process
    videos_to_process = [
        ("../videos_tracking_09/out2.mp4", "out2"),
        ("../videos_tracking_09/out4.mp4", "out4"),
        ("../videos_tracking_09/out13.mp4", "out13")
    ]
    
    # Class names
    class_names = {0: "Ball", 1: "Player", 2: "Referee"}
    
    # Load model (once for all videos)
    print("Loading YOLO model...")
    model = load_model(weights_path)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Process each video
    successful_videos = []
    failed_videos = []
    
    for video_path, video_name in videos_to_process:
        print(f"\nStarting processing for {video_name}...")
        
        success = process_video(
            video_path=video_path,
            video_name=video_name,
            model=model,
            class_names=class_names,
            confidence_threshold=confidence_threshold,
            target_fps=target_fps
        )
        
        if success:
            successful_videos.append(video_name)
        else:
            failed_videos.append(video_name)
    

    
    print(f"\nResults saved in: tracking_2D/results/")


if __name__ == "__main__":
    main()
