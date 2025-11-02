from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np

# Load model
model = YOLO("runs/detect/yolov8_detection/weights/best.pt")

# Initialize tracker
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture("data/sample_videos/people_walk.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("runs/detect/people_counting.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps, (width, height))

# Benzersiz ID'leri saklamak için set
unique_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.4, show=False)

    # Prepare detections for DeepSort
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if cls == 0:  # only person
                detections.append(([x1, y1, x2, y2], conf, cls))

    # Update tracks
    tracks = tracker.update_tracks(detections, frame=frame)

    for t in tracks:
        if not t.is_confirmed():
            continue
        tid = t.track_id
        
        # Yeni ID'yi kaydet
        unique_ids.add(tid)
        
        x1, y1, x2, y2 = map(int, t.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID:{tid}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Şu anki kişi sayısı ve toplam geçen kişi sayısı
    cv2.putText(frame, f"Current: {len(tracks)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Total Passed: {len(unique_ids)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Done. Total people passed: {len(unique_ids)}")
