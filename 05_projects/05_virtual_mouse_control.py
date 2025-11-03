# 05_object_trail_tracker.py
# Tracks a cellphone in real time using YOLOv8 and visualizes its motion trail.
# As the phone moves, a red trail follows its path dynamically.

from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model (lightweight version)
model = YOLO("yolov8n.pt")

# List to store previous positions of the tracked object (the trail)
trail_points = []

# Capture video from the default camera
cap = cv2.VideoCapture(0)

# Class labels in YOLO
class_names = model.names
target_class = "cell phone"  # Only track this object type

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on the frame
    results = model(frame, verbose=False)
    detections = results[0].boxes

    center = None  # Object center point to track

    # Process each detection
    for box in detections:
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id]

        if cls_name == target_class:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, cls_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Add the new center to the trail
    if center:
        trail_points.append(center)

    # Limit the length of the trail to avoid overflow
    if len(trail_points) > 200:
        trail_points.pop(0)

    # Draw the trail line
    for i in range(1, len(trail_points)):
        if trail_points[i - 1] is None or trail_points[i] is None:
            continue
        thickness = int(np.sqrt(200 / float(i + 1)) * 2)
        cv2.line(frame, trail_points[i - 1], trail_points[i], (0, 0, 255), thickness)

    # Display result
    cv2.imshow("Cell Phone Tracker", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
