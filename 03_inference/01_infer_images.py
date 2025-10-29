"""
This script runs inference using a trained YOLOv8 model on sample images.
"""

from ultralytics import YOLO
import os

# Load trained YOLOv8 model (replace with your path if different)
model = YOLO('runs/detect/yolov8_detection/weights/best.pt')

# Path to folder with images
image_folder = 'data/sample_images'
images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png'))]

# Run inference on each image
for img_path in images:
    results = model.predict(
        source=img_path,
        conf=0.4,          # confidence threshold
        save=True,         # save output images with predictions
        show_labels=True,  # display class labels
        show_conf=True     # display confidence scores
    )
    print(f"Inference done for: {img_path}")

print("All images processed.")
