"""
Simplified YOLOv8 fine-tuning script for basketball detection.
Fine-tunes YOLOv8 on basketball dataset to detect: Ball, Player, Referee
"""

from ultralytics import YOLO
import os

def main():
    print("Starting YOLOv8s fine-tuning for basketball detection...")
    
    # Configuration
    data_yaml = "complete_dataset_yolov8/data.yaml"
    epochs = 50
    batch_size = 16
    img_size = 1280
    model_name = "yolov8l.pt"
    
    # Check if data.yaml exists
    if not os.path.exists(data_yaml):
        print(f"Error: Cannot find {data_yaml}")
        print("Please check the path to your dataset configuration file.")
        return
    
    print(f"Configuration:")
    print(f"- Model: {model_name}")
    print(f"- Epochs: {epochs}")
    print(f"- Batch size: {batch_size}")
    print(f"- Image size: {img_size}")
    print(f"- Dataset: {data_yaml}")
    print(f"- Dataset found: ✓")
    
    # Load YOLOv8s model
    print(f"\nLoading {model_name}...")
    model = YOLO(model_name)
    print("Model loaded successfully!")
    
    # Start training
    print("\nStarting training...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        pretrained=True,
        cache=True,
        save=True,
        project="tracking_2D/finetuning_runs/train",  # Save runs folder in current directory (tracking_2D)
        name="basketball_yolov8l",
        exist_ok=True
    )
    
    print("\nTraining completed!")
    print(f"Best weights saved to: {results.save_dir}/weights/best.pt")
    print(f"Last weights saved to: {results.save_dir}/weights/last.pt")
    
    # Validate the model
    print("\nValidating model...")
    val_results = model.val()
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")
    
    print("\nFine-tuning completed successfully!")

if __name__ == "__main__":
    main()
