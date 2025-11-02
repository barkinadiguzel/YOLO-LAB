"""
03_realtime_surveillance.py

Real-time object detection and tracking using YOLOv8 and DeepSort.

- Captures video from webcam or video file.
- Detects objects frame by frame using YOLOv8.
- Tracks objects across frames using DeepSort.
- Displays bounding boxes, IDs, and current number of people.
- Press 'q' to quit.
"""
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("runs/detect/yolov8_detection/weights/best.pt")

# Initialize DeepSort tracker
tracker = DeepSort(max_age=30)

# Video source: 0 for webcam or replace with video file path
cap = cv2.VideoCapture(0)

# Store unique IDs for counting
unique_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
    results = model.predict(frame, conf=0.4, show=False)

    # Prepare detections for DeepSort ([bbox], conf, cls)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if cls == 0:  # only person
                detections.append(([x1, y1, x2, y2], conf, cls))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for t in tracks:
        if not t.is_confirmed():
            continue
        tid = t.track_id
        unique_ids.add(tid)
        x1, y1, x2, y2 = map(int, t.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{tid}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display counts
    cv2.putText(frame, f"Current: {len(tracks)}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Total Passed: {len(unique_ids)}", (10,70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Real-time Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done. Total people passed: {len(unique_ids)}")
