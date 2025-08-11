#!/usr/bin/env python3
"""
Script to convert YOLOv8 label classes from 13 classes to 3 classes.
Mapping rules:
- 0 → 0 (remains the same)
- 1,2,3,4,5,8,9,10,11,12 → 1
- 6,7 → 2
"""

import os
import glob

def convert_class_id(old_class_id):
    """Convert old class ID to new class ID according to mapping rules."""
    if old_class_id == 0:
        return 0
    elif old_class_id in [1, 2, 3, 4, 5, 8, 9, 10, 11, 12]:
        return 1
    elif old_class_id in [6, 7]:
        return 2
    else:
        raise ValueError(f"Unexpected class ID: {old_class_id}")

def process_label_file(file_path):
    """Process a single label file and convert class IDs."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    converted_lines = []
    changes_made = False
    
    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            converted_lines.append(line)
            continue
            
        parts = line.split()
        if len(parts) < 5:  # YOLO format should have at least 5 values
            converted_lines.append(line)
            continue
            
        try:
            old_class_id = int(parts[0])
            new_class_id = convert_class_id(old_class_id)
            
            if old_class_id != new_class_id:
                changes_made = True
                
            # Replace the class ID but keep the rest of the line
            parts[0] = str(new_class_id)
            converted_lines.append(' '.join(parts))
            
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not process line in {file_path}: {line}")
            converted_lines.append(line)
    
    # Write back to file
    with open(file_path, 'w') as f:
        for line in converted_lines:
            f.write(line + '\n')
    
    return changes_made

def main():
    """Main function to process all label files."""
    labels_dir = "dataset_plain_yolov8/train/labels"
    
    if not os.path.exists(labels_dir):
        print(f"Error: Directory {labels_dir} not found!")
        return
    
    # Get all .txt files in the labels directory
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    if not label_files:
        print(f"No .txt files found in {labels_dir}")
        return
    
    print(f"Found {len(label_files)} label files to process...")
    
    files_changed = 0
    total_files = len(label_files)
    
    for i, file_path in enumerate(label_files):
        if (i + 1) % 50 == 0:  # Progress update every 50 files
            print(f"Processing file {i + 1}/{total_files}")
            
        try:
            if process_label_file(file_path):
                files_changed += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"\nConversion completed!")
    print(f"Total files processed: {total_files}")
    print(f"Files with changes: {files_changed}")
    print(f"Files unchanged: {total_files - files_changed}")

if __name__ == "__main__":
    main()
