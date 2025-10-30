"""
Pose estimation using YOLOv8 (keypoint detection).
This script loads a YOLOv8 pose model and runs inference on a sample image.
"""

from ultralytics import YOLO

# Load pretrained YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')  # You can replace with yolov8s-pose.pt for better accuracy

# Run pose estimation on a sample image
results = model.predict(
    source='data/sample_images/sample2.jpg',  # Path to sample image
    conf=0.4,          # Confidence threshold
    save=True,         # Save output image
    show_labels=True,  # Show class labels (like 'person')
    show_conf=True,    # Show confidence scores
)

print("Pose estimation complete. Results saved to:", results[0].save_dir)
