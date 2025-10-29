# Train a YOLOv8 segmentation model on the COCO8-seg dataset

from ultralytics import YOLO

# Load a pretrained YOLOv8 segmentation model (n model = medium)
model = YOLO("yolov8n-seg.pt")

# Train the model on the COCO8-seg dataset for 10 epochs
results = model.train(
    data="coco8-seg.yaml",   # built-in COCO8 segmentation dataset
    epochs=10,
    imgsz=640,               # image size
    project="runs/segment",  # save path
    name="yolov8_segmentation"
)

# Validate and test performance
metrics = model.val()

# Run inference on a sample image
model.predict(source="data/sample_images/sample2.jpg", save=True)

print("Segmentation training and inference complete.")
