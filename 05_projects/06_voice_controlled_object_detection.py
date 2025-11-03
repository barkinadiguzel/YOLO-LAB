# 06_voice_controlled_object_detection.py
# Combines voice recognition and YOLOv8 to perform interactive object detection.
# The system listens to microphone commands like "detect phone" or "stop"
# and adjusts its object tracking behavior accordingly.  I WILL CHANGE IT TODAY

from ultralytics import YOLO
import cv2
import numpy as np
import speech_recognition as sr

# Load YOLO model
model = YOLO("yolov8n.pt")
class_names = model.names

# Initialize speech recognizer
recognizer = sr.Recognizer()
mic = sr.Microphone()

# Video source
cap = cv2.VideoCapture(0)

target_class = None
active = True

print("🎤 Say a command like 'detect phone' or 'stop'...")

def listen_for_command():
    global target_class, active
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=4)
            command = recognizer.recognize_google(audio).lower()
            print(f"Command heard: {command}")

            if "detect" in command:
                for name in class_names.values():
                    if name in command:
                        target_class = name
                        print(f"✅ Target set to: {target_class}")
                        active = True
                        return
                print("❌ Object not recognized.")
            elif "stop" in command:
                active = False
                print("🛑 Detection stopped.")
            elif "resume" in command:
                active = True
                print("▶️ Detection resumed.")
            else:
                print("⚠️ Unrecognized command.")
        except Exception as e:
            print(f"Listening error: {e}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    if active and target_class:
        results = model(frame, verbose=False)
        detections = results[0].boxes

        for box in detections:
            cls_id = int(box.cls[0])
            cls_name = class_names[cls_id]

            if cls_name == target_class:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, cls_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"Target: {target_class or 'None'}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Voice-Controlled YOLO", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('v'):  # press 'v' to listen for a new command
        listen_for_command()

cap.release()
cv2.destroyAllWindows()
