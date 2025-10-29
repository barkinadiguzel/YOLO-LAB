"""
This script evaluates a trained YOLOv8 model on a test dataset or sample images.
It prints key metrics (mAP, precision, recall) and optionally saves inference results.
"""

from ultralytics import YOLO
import os

# ==============================
# 1. Load the trained YOLO model
# ==============================
# Update this path to your trained model's weights
model_path = 'runs/detect/yolov8_detection/weights/best.pt'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found at: {model_path}\n"
                            "Train the model first or update the path.")

model = YOLO(model_path)

# ==============================
# 2. Evaluate on a test dataset (optional)
# ==============================
# If you have a dataset yaml for validation/testing
data_yaml = 'data/coco8.yaml'  # Update if you have a custom dataset

if os.path.exists(data_yaml):
    metrics = model.val(data=data_yaml)
    print("Evaluation metrics on test dataset:")
    print(metrics)
else:
    print(f"No dataset YAML found at {data_yaml}. Skipping full dataset evaluation.")

# ==============================
# 3. Run inference on a sample image
# ==============================
sample_image = 'data/sample_images/sample1.jpg'  # Update or add more images
if os.path.exists(sample_image):
    results = model.predict(
        source=sample_image,
        conf=0.4,
        save=True,
        show_labels=True,
        show_conf=True
    )
    print(f"Inference complete. Results saved to {results[0].save_dir}")
else:
    print(f"Sample image not found at {sample_image}. Skipping inference.")
