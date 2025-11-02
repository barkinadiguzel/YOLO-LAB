"""
02_people_counting.py

Counts people in a video using a YOLOv8 model.

- Loads a YOLOv8 model.
- Reads a video frame by frame.
- Detects people and draws bounding boxes.
- Displays the number of people on each frame.
- Press 'q' to quit.
"""
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("runs/detect/yolov8_detection/weights/best.pt")

# Video source
video_path = "data/sample_videos/people_walk.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict objects in the frame
    results = model.predict(frame, conf=0.4, show=False)

    # Count people
    person_count = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == "person":
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show count on frame
    cv2.putText(frame, f"People: {person_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("People Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
