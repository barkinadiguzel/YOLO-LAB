"""
This script runs real-time inference on a webcam feed using a trained YOLOv8 model.
"""

from ultralytics import YOLO

# Load trained YOLOv8 model
model = YOLO('runs/detect/yolov8_detection/weights/best.pt')

# Run inference on webcam (0 = default camera)
results = model.predict(
    source=0,          # webcam device index
    conf=0.4,          # confidence threshold
    show=True,         # show live predictions
    save=False,        # optional: can save video if needed
    show_labels=True,
    show_conf=True
)

print("Webcam inference complete.")
