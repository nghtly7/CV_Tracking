import cv2
import os
import re
import numpy as np
from PIL import Image
import json
from pathlib import Path

def find_exact_matching_frame(annot_path, video_path, max_search=20, verbose=True, resize=True):
    """
    Returns the 1-based frame number in video that exactly matches the annotated image,
    or the closest frame among the first `max_search` frames if no exact match.
    """
    ann_img = cv2.imread(annot_path)
    if ann_img is None:
        raise FileNotFoundError(f"Annotation image not found: {annot_path}")
    ann_img = cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB)
    h_ann, w_ann = ann_img.shape[:2]

    cap = cv2.VideoCapture(video_path)
    best_frame_idx = None
    best_diff = float('inf')
    exact_match = None

    if verbose:
        print(f"[INFO] Searching for exact match in first {max_search} frames of {video_path} for {annot_path}...")

    for i in range(1, max_search + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if resize and frame_rgb.shape[:2] != (h_ann, w_ann):
            frame_rgb = cv2.resize(frame_rgb, (w_ann, h_ann))

        diff = np.sum(np.abs(frame_rgb.astype(np.int32) - ann_img.astype(np.int32)))
        if diff < best_diff:
            best_diff = diff
            best_frame_idx = i

        if diff == 0:
            exact_match = i
            if verbose:
                print(f"[RESULT] Exact pixel-wise match found at video frame {i}")
            break

        if verbose:
            print(f"[INFO] Checked frame {i}... Current best: frame {best_frame_idx} (difference {best_diff})")

    cap.release()
    
    # Return the best match
    match_idx = exact_match if exact_match is not None else best_frame_idx
    if verbose:
        if exact_match is None:
            print(f"[FINAL] Closest frame in first {max_search}: {best_frame_idx} (pixel sum difference: {best_diff})")
        else:
            print(f"[FINAL] Exact match found at frame {exact_match}")
    
    return match_idx, best_diff

def extract_video_id_from_filename(filename):
    """Extract the video ID from a filename"""
    # First try to match pattern like 'out13_frame_0001_png...'
    match = re.search(r'([^_]+)_frame_', filename)
    if match:
        return match.group(1)
    
    # Try alternative pattern like 'out13_0001...'
    match = re.search(r'([^_]+)_\d+', filename)
    if match:
        return match.group(1)
    
    # If no match, use the name without extension
    return Path(filename).stem.split('_')[0]

def find_ground_truth_image(gt_folder, video_id):
    """Find the first ground truth image for the specified video ID"""
    # Look for images with the video_id prefix in the folder
    for file in sorted(os.listdir(gt_folder)):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')) and file.startswith(video_id):
            print(f"Found ground truth image: {file}")
            return os.path.join(gt_folder, file)
    
    raise FileNotFoundError(f"No ground truth image found for video ID: {video_id}")

def check_frame_coordination(video_path, gt_folder="GroundTruthData/train/"):
    """Main function to check frame coordination between video and ground truth"""
    # Extract video ID from video path
    video_filename = os.path.basename(video_path)
    video_id = os.path.splitext(video_filename)[0]
    
    print(f"Checking coordination for video: {video_id}")
    
    # Find first ground truth image for this video
    try:
        gt_image_path = find_ground_truth_image(gt_folder, video_id)
    except FileNotFoundError:
        # If not found with exact name, try to find by checking all ground truth images
        print(f"No direct match found. Trying to infer from ground truth image names...")
        
        # Get all ground truth image file names
        gt_files = [f for f in os.listdir(gt_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not gt_files:
            raise FileNotFoundError(f"No ground truth images found in {gt_folder}")
        
        # Extract unique video IDs from ground truth files
        gt_video_ids = set(extract_video_id_from_filename(f) for f in gt_files)
        print(f"Found ground truth for these video IDs: {gt_video_ids}")
        
        # Check if any ground truth video ID is contained in our video filename
        matching_ids = [gid for gid in gt_video_ids if gid in video_id]
        
        if matching_ids:
            closest_id = max(matching_ids, key=len)  # Take the longest matching ID
            print(f"Found closest matching video ID: {closest_id}")
            gt_image_path = find_ground_truth_image(gt_folder, closest_id)
        else:
            raise FileNotFoundError(f"Could not find any ground truth images for {video_id}")
    
    # Use the find_exact_matching_frame function to find the matching frame
    match_index, diff_value = find_exact_matching_frame(
        annot_path=gt_image_path, 
        video_path=video_path,
        max_search=20,  # First 20 frames
        verbose=True,
        resize=True
    )
    
    # Display results
    print("\nFrame Matching Results:")
    print(f"Best matching frame: #{match_index}")
    print(f"Pixel difference: {diff_value}")
    
    if diff_value == 0:
        print("Perfect pixel-wise match found!")
    else:
        print("No perfect match found. Using closest frame.")
    
    # Return the frame offset
    return match_index

if __name__ == "__main__":
    # Example usage
    video_path = "raw_video/out13.mp4"  # Change this to your video path
    gt_folder = "GroundTruthData/train/"  # Ground truth folder
    
    frame_offset = check_frame_coordination(video_path, gt_folder)
    print(f"\nConclusion: Video starts at frame #{frame_offset} relative to the ground truth")
    
    # Save the result
    result = {
        "video_path": video_path,
        "frame_offset": frame_offset,
        "frame_indexing": "1-based"  # Explicitly indicate the indexing convention
    }
    
    

