import cv2
import os

def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
        if frame_count % 100 == 0: # Stampa un messaggio ogni 100 frame
            print(f"Extracted {frame_count} frames from {video_path}")

    cap.release()
    print(f"Finished extracting frames from {video_path}. Total: {frame_count}")

# Esempio di utilizzo per le 3 angolazioni
# extract_frames('path/to/video_angle1.mp4', 'frames/angle1/')
# extract_frames('path/to/video_angle2.mp4', 'frames/angle2/')
# extract_frames('path/to/video_angle3.mp4', 'frames/angle3/')

if __name__ == "__main__":
    extract_frames('raw_video/out2.mp4', 'frames/out2/')
    extract_frames('raw_video/out4.mp4', 'frames/out4/')
    extract_frames('raw_video/out13.mp4', 'frames/out13/')