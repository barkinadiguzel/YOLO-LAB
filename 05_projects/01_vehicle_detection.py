"""
01_vehicle_detection.py

A simple YOLOv8 script for vehicle detection using a pretrained model.

This script:
1. Loads a pretrained YOLOv8 model
2. Runs detection on an input image or video
3. Saves and optionally displays the results

You can replace the model or source with your own data.
"""

from ultralytics import YOLO

# Load a pretrained YOLOv8 model (you can use yolov8n.pt for speed or yolov8m.pt for accuracy)
model = YOLO("yolov8n.pt")

# Run detection on an image or video
results = model.predict(
    source="data/sample_images/traffic.jpg",  # Path to your test image or video
    conf=0.4,                                # Confidence threshold
    save=True,                               # Save output images/videos
    show=True                                # Display result window (optional)
)

print("Vehicle detection complete. Results saved in 'runs/detect' folder.")
