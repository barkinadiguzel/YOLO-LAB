"""
Export a trained YOLOv8 detection model to ONNX format.

This script:
1. Loads a fine-tuned YOLOv8 model (.pt file)
2. Converts it to ONNX format for cross-platform deployment
3. Saves the exported model inside the 'exports' directory

The ONNX format allows integration with frameworks like:
- ONNX Runtime (C++, Python, Android, iOS)
- OpenCV DNN module
- TensorRT or other inference engines
"""

from ultralytics import YOLO
import os

# Path to the trained YOLOv8 model (.pt file)
model_path = r"runs\detect\yolov8_detection\weights\best.pt"

# Load the YOLO model
model = YOLO(model_path)

# Create the output directory for exported models
os.makedirs("exports", exist_ok=True)

print("Exporting model to ONNX format...")

# Export the model to ONNX
onnx_path = model.export(format="onnx", opset=12, dynamic=True, simplify=True)

print(f"ONNX export complete: {onnx_path}")
