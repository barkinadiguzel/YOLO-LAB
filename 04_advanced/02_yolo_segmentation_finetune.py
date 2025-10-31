"""
Fine-tune a YOLOv8 segmentation model on a custom dataset.
This example uses the COCO8 segmentation dataset.
"""

from ultralytics import YOLO

# Load a pre-trained YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')  # nano segmentation model

# Train / Fine-tune the model
results = model.train(
    data='data/coco8-seg.yaml',  # dataset config with segmentation masks
    epochs=10,
    imgsz=640,
    batch=2,
    device='cpu',                # change to '0' if GPU available
    project='runs/segment',
    name='yolov8_segmentation'
)

# Load the trained weights
trained_model = YOLO('runs/segment/yolov8_segmentation5/weights/best.pt')

# Optional: test on a sample image
results = trained_model.predict(
    source='data/sample_images/sample1.jpg',
    conf=0.4,
    save=True,
    show_labels=True,
    show_conf=True
)

print("Segmentation training and inference complete.")
