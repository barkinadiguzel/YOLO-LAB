"""
Load a pre-trained YOLOv8 model and perform a simple inference to verify setup.

This script demonstrates how to:
1. Load a YOLOv8 detection or segmentation model.
2. Run it on a sample image.
3. Check if the output works without errors.
"""

from ultralytics import YOLO

# Load pre-trained YOLOv8 model (detection)
model = YOLO('yolov8n.pt')  # you can change to yolov8n-seg.pt for segmentation

# Run a quick inference on a sample image
results = model.predict('data/sample_images/sample1.jpg', conf=0.4, save=False)

# Print summary of detections
print(results)
