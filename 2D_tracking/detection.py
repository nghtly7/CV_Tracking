from ultralytics import YOLO
import cv2
import os
import json

# Config
video_path = "../rectified_video/out13.mp4"
model_path = "best.pt"
video_name = os.path.splitext(os.path.basename(video_path))[0]
output_dir = "rDetection"
output_json = os.path.join(output_dir, f"{video_name}_detections.json")
output_video = os.path.join(output_dir, f"{video_name}_annotated.mp4")
conf_threshold = 0.3

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

# Get video properties for output video
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create video writer for annotated output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

frame_id = 0

all_detections = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=conf_threshold)[0]
    frame_detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        frame_detections.append([x1, y1, w, h, conf, cls])
        
        # Draw bounding box on frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Add confidence and class label
        label = f"Class {cls}: {conf:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Write annotated frame to output video
    out.write(frame)

    all_detections.append({
        "frame": frame_id,
        "detections": frame_detections
    })

    frame_id += 1

cap.release()
out.release()

# Salva tutto in un solo file
with open(output_json, "w") as f:
    json.dump(all_detections, f, indent=2)

print(f"[✓] Detection salvate in: {output_json}")
print(f"[✓] Video annotato salvato in: {output_video}")
