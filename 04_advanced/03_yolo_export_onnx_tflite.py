"""
03_yolo_export_onnx_tflite.py
Export a trained YOLOv8 model to ONNX and TFLite formats.
"""

from ultralytics import YOLO
import os

# Load trained YOLO model
model = YOLO("runs/detect/yolov8_detection/weights/best.pt")

# Create export directory
os.makedirs("exports", exist_ok=True)

# Export to ONNX
model.export(format="onnx", imgsz=640)
print("Exported to ONNX format.")

# Export to TFLite
model.export(format="tflite", imgsz=640)
print("Exported to TFLite format.")
