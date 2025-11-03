"""
05_object_trail_tracker.py
--------------------------------
 Real-Time Phone Tracker with Trail

This code uses a YOLOv8 model to detect a phone in a live camera feed.
When the phone is detected, it draws a bounding box around it and
tracks its movement across frames. The phone’s path is visualized
as a trail that follows its previous positions on the screen.

Basically: move your phone in front of the camera, and you’ll see a
colored line showing where it’s been — like a motion trace.
"""

from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Start webcam
cap = cv2.VideoCapture(0)

# List to store trail points
trail_points = []
MAX_TRAIL_LENGTH = 30  # Limit trail length

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, verbose=False)

    # Process detections
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Only track phone
            if label == "cell phone":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Store trail points
                trail_points.append((cx, cy))
                if len(trail_points) > MAX_TRAIL_LENGTH:
                    trail_points.pop(0)

                # Draw phone box and center
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # Draw trail line
    for i in range(1, len(trail_points)):
        cv2.line(frame, trail_points[i - 1], trail_points[i], (255, 0, 0), 2)

    # Label text
    cv2.putText(frame, "Tracking: Cell Phone", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Phone Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
