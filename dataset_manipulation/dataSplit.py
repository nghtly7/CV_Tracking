import os
import shutil
import random
from pathlib import Path

def create_directories(base_path):
    """Create necessary directories if they don't exist"""
    directories = [
        os.path.join(base_path, 'train', 'images'),
        os.path.join(base_path, 'train', 'labels'),
        os.path.join(base_path, 'test', 'images'),
        os.path.join(base_path, 'test', 'labels'),
        os.path.join(base_path, 'valid', 'images'),
        os.path.join(base_path, 'valid', 'labels')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created/verified: {directory}")

def get_image_files(images_path):
    """Get all image files from the source directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for file in os.listdir(images_path):
        if Path(file).suffix.lower() in image_extensions:
            image_files.append(file)
    
    return image_files

def move_files(file_list, source_images_path, source_labels_path, dest_images_path, dest_labels_path):
    """Move image and corresponding label files to destination"""
    moved_count = 0
    
    for image_file in file_list:
        # Move image file
        source_image = os.path.join(source_images_path, image_file)
        dest_image = os.path.join(dest_images_path, image_file)
        
        if os.path.exists(source_image):
            shutil.move(source_image, dest_image)
            moved_count += 1
            
            # Move corresponding label file (change extension to .txt)
            label_file = Path(image_file).stem + '.txt'
            source_label = os.path.join(source_labels_path, label_file)
            dest_label = os.path.join(dest_labels_path, label_file)
            
            if os.path.exists(source_label):
                shutil.move(source_label, dest_label)
                print(f"Moved: {image_file} and {label_file}")
            else:
                print(f"Warning: Label file not found for {image_file}")
        
    return moved_count

def split_dataset(base_path="complete_dataset_yolov8", train_ratio=0.8, test_ratio=0.1, valid_ratio=0.1):
    """
    Split dataset into train, test, and validation sets
    
    Args:
        base_path: Path to the dataset directory
        train_ratio: Percentage for training set (default 0.8 = 80%)
        test_ratio: Percentage for test set (default 0.1 = 10%)
        valid_ratio: Percentage for validation set (default 0.1 = 10%)
    """
    
    # Verify ratios sum to 1
    if abs(train_ratio + test_ratio + valid_ratio - 1.0) > 0.001:
        raise ValueError("Train, test, and validation ratios must sum to 1.0")
    
    # Define paths
    source_images_path = os.path.join(base_path, 'train', 'images')
    source_labels_path = os.path.join(base_path, 'train', 'labels')
    
    # Check if source directory exists
    if not os.path.exists(source_images_path):
        raise FileNotFoundError(f"Source images directory not found: {source_images_path}")
    
    # Create destination directories
    create_directories(base_path)
    
    # Get all image files
    all_images = get_image_files(source_images_path)
    
    if not all_images:
        print("No image files found in the source directory!")
        return
    
    print(f"Found {len(all_images)} images to split")
    
    # Shuffle the list for random distribution
    random.shuffle(all_images)
    
    # Calculate split indices
    total_images = len(all_images)
    train_end = int(total_images * train_ratio)
    test_end = train_end + int(total_images * test_ratio)
    
    # Split the file list
    train_files = all_images[:train_end]
    test_files = all_images[train_end:test_end]
    valid_files = all_images[test_end:]
    
    print(f"Splitting into:")
    print(f"  Train: {len(train_files)} images ({len(train_files)/total_images*100:.1f}%)")
    print(f"  Test: {len(test_files)} images ({len(test_files)/total_images*100:.1f}%)")
    print(f"  Valid: {len(valid_files)} images ({len(valid_files)/total_images*100:.1f}%)")
    
    # Move test files
    if test_files:
        test_images_path = os.path.join(base_path, 'test', 'images')
        test_labels_path = os.path.join(base_path, 'test', 'labels')
        moved_test = move_files(test_files, source_images_path, source_labels_path, 
                               test_images_path, test_labels_path)
        print(f"Moved {moved_test} images to test set")
    
    # Move validation files
    if valid_files:
        valid_images_path = os.path.join(base_path, 'valid', 'images')
        valid_labels_path = os.path.join(base_path, 'valid', 'labels')
        moved_valid = move_files(valid_files, source_images_path, source_labels_path,
                                valid_images_path, valid_labels_path)
        print(f"Moved {moved_valid} images to validation set")
    
    # Remaining files stay in train (they're already there)
    print(f"Remaining {len(train_files)} images left in training set")
    print("Dataset split completed successfully!")

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    
    print("Starting dataset split...")
    print("This will move 10% of images to test/ and 10% to valid/")
    print("The remaining 80% will stay in train/")
    
    try:
        split_dataset()
    except Exception as e:
        print(f"Error: {e}")