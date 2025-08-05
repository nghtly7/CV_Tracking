import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import motmetrics as mm
from pathlib import Path
import torch
from ultralytics import YOLO
import supervision as sv


# Define class labels
CLASS_NAMES = [
    "ball", "red_0", "red_11", "red_12", "red_16", "red_2", 
    "ref_f", "ref_m", "white_13", "white_16", "white_25", "white_27", "white_34"
]

# Create a mapping from class ID to class name
CLASS_ID_TO_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}


def fine_tune_yolov8(data_yaml, epochs=50, img_size=640, batch_size=16):
    """
    Fine-tune YOLOv8 model on custom dataset
    
    Args:
        data_yaml: Path to the data YAML file
        epochs: Number of training epochs
        img_size: Input image size
        batch_size: Batch size for training
        
    Returns:
        Path to the best weights file
    """
    # Load a pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')  # Using nano model as base, can be changed to s/m/l/x
    
    # Fine-tune the model on custom dataset
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name='yolov8_fine_tuned'
    )
    
    # Return path to best weights
    return str(Path('runs/detect/yolov8_fine_tuned/weights/best.pt'))


def create_tracker():
    """
    Create a ByteTrack tracker instance
    
    Returns:
        ByteTrack tracker instance
    """
    # Create ByteTrack tracker
    tracker = sv.ByteTrack()
    
    return tracker


def run_tracking(video_path, model_path, output_file, view_id, conf_threshold=0.25, iou_threshold=0.7):
    """
    Run 2D tracking on a video using YOLOv8 and ByteTrack
    
    Args:
        video_path: Path to the input video
        model_path: Path to the YOLOv8 model weights
        output_file: Path to save tracking results
        view_id: ID of the camera view
        conf_threshold: Confidence threshold for detections
        iou_threshold: IOU threshold for NMS
        
    Returns:
        DataFrame with tracking results
    """
    # Load the YOLOv8 model
    model = YOLO(model_path)
    
    # Create tracker
    tracker = create_tracker()
    
    # Create annotator for visualization
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer for visualization
    video_output_path = output_file.replace('.csv', '.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
    
    # Prepare output results
    output_results = []
    
    # Process each frame
    with tqdm(total=total_frames, desc=f"Processing video {view_id}") as pbar:
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLOv8 detection
            results = model(frame, conf=conf_threshold, iou=iou_threshold)[0]
            
            # Convert detections to supervision format
            detections = sv.Detections.from_ultralytics(results)
            
            # Update tracker
            detections = tracker.update_with_detections(detections)
            
            # Inside the while loop, after updating tracker:
            # Save tracking results
            if len(detections) > 0:
                for xyxy, mask, confidence, class_id, tracker_id, data in detections:
                    # Get class name
                    class_name = CLASS_ID_TO_NAME.get(int(class_id), f"unknown_{class_id}")
                    
                    # Get bounding box coordinates directly from xyxy
                    x1, y1, x2, y2 = xyxy
                    
                    # Convert to x, y, w, h format
                    x = x1
                    y = y1
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Save results with class name
                    output_results.append([
                        frame_id, tracker_id, x, y, w, h, 
                        confidence, class_id, 1.0, -1, view_id, class_name
                    ])
            
            # Visualize and save to video
            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
            
            # Add labels with tracker IDs and class names
            labels = [
                f"#{tracker_id} {CLASS_ID_TO_NAME.get(int(class_id), f'unknown_{class_id}')} {confidence:.2f}"
                for _, _, confidence, class_id, tracker_id, _ in detections
            ]
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            
            # Write frame to video
            video_writer.write(annotated_frame)
            
            # Optional: Display frame
            # cv2.imshow("Tracking", annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    
    # Don't forget to release the video writer
    video_writer.release()
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save results to CSV
    result_df = pd.DataFrame(output_results, columns=[
        'frame_id', 'id', 'x', 'y', 'w', 'h', 'confidence', 'class_id', 
        'visibility', 'unused', 'view_id', 'class_name'
    ])
    result_df.to_csv(output_file, index=False)
    
    return result_df


def evaluate_tracking(gt_file, result_file):
    """
    Evaluate tracking performance using standard metrics
    
    Args:
        gt_file: Path to ground truth file
        result_file: Path to tracking results file
        
    Returns:
        Summary of evaluation metrics
    """
    # Load ground truth and results
    # When loading results, ensure it works with or without the class_name column
    gt = pd.read_csv(gt_file)
    results = pd.read_csv(result_file)
    
    # Initialize the metrics accumulator
    acc = mm.MOTAccumulator(auto_id=True)
    
    # Group by frame_id
    gt_by_frame = gt.groupby('frame_id')
    results_by_frame = results.groupby('frame_id')
    
    # Process each frame
    for frame_id in gt_by_frame.groups:
        if frame_id not in results_by_frame.groups:
            continue
            
        gt_frame = gt_by_frame.get_group(frame_id)
        results_frame = results_by_frame.get_group(frame_id)
        
        # Extract IDs and positions
        gt_ids = gt_frame['id'].values
        result_ids = results_frame['id'].values
        
        # Calculate distances between bounding boxes
        gt_boxes = gt_frame[['x', 'y', 'w', 'h']].values
        result_boxes = results_frame[['x', 'y', 'w', 'h']].values
        
        distances = mm.distances.iou_matrix(gt_boxes, result_boxes, max_iou=0.5)
        
        # Update the accumulator
        acc.update(gt_ids, result_ids, distances)
    
    # Calculate metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=[
        'mota', 'motp', 'idf1', 'num_switches', 'num_fragmentations'
    ])
    
    return summary


def visualize_tracking_results(result_file, video_path=None):
    """
    Visualize tracking results with class labels
    
    Args:
        result_file: Path to tracking results CSV
        video_path: Optional path to original video for background
    """
    results = pd.read_csv(result_file)
    
    # Plot trajectories by class
    plt.figure(figsize=(12, 10))
    
    # Get unique classes
    unique_classes = results['class_name'].unique() if 'class_name' in results.columns else [f"class_{i}" for i in results['class_id'].unique()]
    
    # Create color map for classes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
    class_colors = {cls: color for cls, color in zip(unique_classes, colors)}
    
    # Plot each object's trajectory
    for obj_id in results['id'].unique():
        obj_data = results[results['id'] == obj_id]
        
        # Get class for this object (use most common class)
        if 'class_name' in obj_data.columns:
            obj_class = obj_data['class_name'].value_counts().index[0]
        else:
            obj_class = f"class_{obj_data['class_id'].value_counts().index[0]}"
        
        color = class_colors[obj_class]
        plt.plot(obj_data['x'], obj_data['y'], '-', color=color, linewidth=1, alpha=0.7)
        plt.plot(obj_data['x'].iloc[0], obj_data['y'].iloc[0], 'o', color=color, markersize=5, label=f"{obj_class} #{obj_id}")
    
    # Create legend with unique entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.1, 1))
    
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Object Trajectories by Class')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the visualization
    output_path = result_file.replace('.csv', '_trajectories.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Visualization saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='2D Tracking with YOLOv8 and ByteTrack')
    parser.add_argument('--data_dir', type=str, default='tracking_09_dataset/train', 
                        help='Directory containing the dataset')
    parser.add_argument('--video_dir', type=str, default='tracking_09', 
                        help='Directory containing the videos')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Directory to save results')
    parser.add_argument('--train', action='store_true', 
                        help='Fine-tune YOLOv8 model')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                        help='Path to YOLOv8 model weights')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=640, 
                        help='Image size for training and inference')
    parser.add_argument('--conf', type=float, default=0.25, 
                        help='Confidence threshold for detections')
    parser.add_argument('--iou', type=float, default=0.7, 
                        help='IOU threshold for NMS')
    parser.add_argument('--view_id', type=int, default=1, 
                        help='ID of the camera view')
    parser.add_argument('--evaluate', action='store_true', 
                        help='Evaluate tracking results')
    parser.add_argument('--gt', type=str, help='Path to ground truth file for evaluation')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Fine-tune YOLOv8 model if requested
    if args.train:
        print("Fine-tuning YOLOv8 model...")
        data_yaml = os.path.join(args.data_dir, 'data.yaml')
        model_path = fine_tune_yolov8(
            data_yaml=data_yaml,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch_size
        )
    else:
        model_path = args.model
    
    # Get list of videos
    video_files = [f for f in os.listdir(args.video_dir) if f.endswith(('.mp4', '.avi'))]
    
    # After processing each video
    for video_file in video_files:
        video_path = os.path.join(args.video_dir, video_file)
        output_file = os.path.join(args.output_dir, f"{os.path.splitext(video_file)[0]}_tracking.csv")
        
        print(f"Processing video: {video_file}")
        results = run_tracking(
            video_path=video_path,
            model_path=model_path,
            output_file=output_file,
            view_id=args.view_id,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        
        # Visualize tracking results
        print(f"Visualizing tracking results for {video_file}...")
        visualize_tracking_results(output_file, video_path)
        
        # Evaluate tracking if requested
        if args.evaluate and args.gt:
            print(f"Evaluating tracking results for {video_file}...")
            summary = evaluate_tracking(args.gt, output_file)
            print(f"Evaluation results for {video_file}:")
            print(summary)


if __name__ == "__main__":
    main()