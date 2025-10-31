"""
Fine-tune a YOLOv8 segmentation model on a custom dataset.

This script loads a pre-trained segmentation model (yolov8n-seg.pt)
and fine-tunes it using your dataset configuration (custom_seg.yaml).
Make sure your dataset is properly structured in YOLO format.

NOTE:
The custom segmentation dataset is NOT included in this repository.
You must provide your own dataset and configuration file (e.g., data/custom_seg.yaml)
before running this script.
"""

from ultralytics import YOLO

# Load pre-trained segmentation model
model = YOLO('yolov8n-seg.pt')

# Fine-tune the model on a custom dataset
results = model.train(
    data='data/custom_seg.yaml',   # Path to your dataset YAML
    epochs=20,                     # Number of epochs
    imgsz=640,                     # Image size
    batch=4,                       # Batch size
    device='cpu',                  # Use 'cuda' if GPU available
    project='runs/segment',        # Folder for saving runs
    name='custom_seg_finetune'     # Name of this training run
)

# Load the best fine-tuned model
fine_tuned_model = YOLO('runs/segment/custom_seg_finetune/weights/best.pt')

# Test on a sample image
results = fine_tuned_model.predict(
    source='data/sample_images/sample2.jpg',  # Example test image
    conf=0.4,
    save=True,
    show_labels=True,
    show_conf=True
)

print("Segmentation fine-tuning and prediction complete.")
