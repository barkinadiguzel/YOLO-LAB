"""
Real-time YOLOv8 demo using webcam or any video capture device.
"""

from ultralytics import YOLO
import cv2

# Load trained YOLOv8 model
model = YOLO('runs/detect/yolov8_detection/weights/best.pt')

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Make sure it's connected.")

print("Starting real-time demo. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting.")
        break

    # Run inference on the frame
    results = model.predict(
        source=frame,
        conf=0.4,
        save=False,
        show=True,         # OpenCV window shows predictions
        show_labels=True,
        show_conf=True
    )

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Real-time demo ended.")
