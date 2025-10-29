"""
This script runs inference on videos using a trained YOLOv8 model.
"""

from ultralytics import YOLO
import os

# Load trained YOLOv8 model
model = YOLO('runs/detect/yolov8_detection/weights/best.pt')

# Path to folder with videos
video_folder = 'data/sample_videos'
videos = [os.path.join(video_folder, v) for v in os.listdir(video_folder) if v.endswith(('.mp4', '.avi'))]

# Run inference on each video
for vid_path in videos:
    results = model.predict(
        source=vid_path,
        conf=0.4,       # confidence threshold
        save=True,      # save output video with predictions
        show_labels=True,
        show_conf=True
    )
    print(f"Inference done for: {vid_path}")

print("All videos processed.")
