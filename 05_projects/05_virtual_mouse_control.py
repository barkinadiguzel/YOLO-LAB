# 05_object_trail_tracker.py
# Telefonu takip eder, hareket ettikçe arkasında iz bırakır.

from ultralytics import YOLO
import cv2
import numpy as np

# 1. YOLO modelini yükle
model = YOLO("yolov8n.pt")  # Küçük ve hızlı model

# 2. Trail (iz) için liste oluştur
trail_points = []

# 3. Video kaynağı (kamera)
cap = cv2.VideoCapture(0)

# 4. YOLO sınıf isimleri
class_names = model.names
target_class = "cell phone"  # Sadece bu nesneyi takip edeceğiz

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    detections = results[0].boxes

    # Ekrana çizilecek merkez noktası
    center = None

    for box in detections:
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id]

        if cls_name == target_class:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Nesne kutusunu çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, cls_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Trail çizimi
    if center:
        trail_points.append(center)

    # Maksimum trail uzunluğu (daha uzun istersen 1000 yap)
    if len(trail_points) > 200:
        trail_points.pop(0)

    # İz çiz
    for i in range(1, len(trail_points)):
        if trail_points[i - 1] is None or trail_points[i] is None:
            continue
        thickness = int(np.sqrt(200 / float(i + 1)) * 2)
        cv2.line(frame, trail_points[i - 1], trail_points[i], (0, 0, 255), thickness)

    cv2.imshow("Cell Phone Tracker", frame)

    # Çıkmak için 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
