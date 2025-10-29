"""
This script demonstrates how to train a YOLOv8 detection model
on a **custom dataset** either from scratch or using pretrained weights.

NOTE:
The custom dataset used in this example (custom_dataset.yaml and its images)
is NOT included in this repository.
You need to create or provide your own dataset and update the data path accordingly.

- Option 1 (recommended): Fine-tuning from pretrained model (faster & better results)
- Option 2: Training from scratch (no pretrained weights)
"""

from ultralytics import YOLO

# ==============================
# Option 1: Fine-tuning from pretrained model
# ==============================
# model = YOLO('yolov8n.pt')  # Loads pretrained weights (recommended for most cases)

# ==============================
# Option 2: Training from scratch
# ==============================
model = YOLO('yolov8n.yaml')  # Starts training from random weights

# ==============================
# Train on your custom dataset
# ==============================
results = model.train(
    data='data/custom_dataset.yaml',  # Path to your dataset config file (not included in repo)
    epochs=20,                        # Adjust as needed
    imgsz=640,                        # Image size
    batch=4,                          # Batch size (depends on your system)
    device='cpu',                     # Change to '0' if GPU is available
    project='runs/custom_train',      # Output directory
    name='custom_yolo_training'       # Experiment name
)

# ==============================
# Optional: Test the trained model on a sample image
# ==============================
trained_model = YOLO('runs/custom_train/custom_yolo_training/weights/best.pt')
results = trained_model.predict(
    source='data/sample_images/sample2.jpg',
    conf=0.4,
    save=True
)

print("Custom dataset training and testing complete.")
