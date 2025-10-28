"""
This script trains a YOLOv8 detection model on the COCO8 dataset.
The model will learn to detect objects defined in coco8.yaml for 10 epochs.
"""

from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (nano version)
model = YOLO('yolov8n.pt')

# Train the model on COCO8 dataset
results = model.train(
    data='data/coco8.yaml',  # Path to dataset config
    epochs=10,               # Number of training epochs
    imgsz=640,               # Image size
    batch=2,                 # Batch size
    device='cpu',           # Use GPU if available
    project='runs/detect',   # Folder to save training outputs
    name='yolov8_detection'  # Name of this training run
)

# Load the trained model (best weights)
trained_model = YOLO('runs/detect/yolov8_detection/weights/best.pt')

# Optional: test on a sample image
results = trained_model.predict(
    source='data/sample_images/sample1.jpg',
    conf=0.4,
    save=True,
    show_labels=True,
    show_conf=True
)

print("Detection training and inference complete.")
