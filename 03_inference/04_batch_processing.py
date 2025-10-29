"""
Real batch inference for multiple images using a trained YOLOv8 model.
"""

from ultralytics import YOLO
import os

# Load trained YOLOv8 model
model = YOLO('runs/detect/yolov8_detection/weights/best.pt')

# Directory containing images
input_dir = 'data/sample_images/'
output_dir = 'runs/detect/batch_predict/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Run batch inference on all images in the folder
results = model.predict(
    source=input_dir,   # folder path
    conf=0.4,
    batch=4,            # batch size
    save=True,
    save_dir=output_dir,
    show=False,
    show_labels=True,
    show_conf=True
)

print("Batch image inference complete.")
