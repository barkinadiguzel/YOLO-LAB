# 06_voice_controlled_object_detection.py
# Detect objects in real-time using YOLOv8 and switch targets via voice commands

from ultralytics import YOLO
import cv2
import speech_recognition as sr

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Default target
target_class = "cell phone"

# Initialize speech recognizer
recognizer = sr.Recognizer()
mic = sr.Microphone()

def listen_for_command():
    global target_class
    try:
        with mic as source:
            print("Listening for command...")
            audio = recognizer.listen(source, timeout=3)
        command = recognizer.recognize_google(audio).lower()
        if "phone" in command:
            target_class = "cell phone"
        elif "bottle" in command:
            target_class = "bottle"
        elif "person" in command:
            target_class = "person"
        print(f"Target changed to: {target_class}")
    except:
        pass  # ignore if nothing understood

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model(frame, verbose=False)
    detections = results[0].boxes

    for box in detections:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        if cls_name == target_class:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, cls_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Voice Controlled Object Detection", frame)

    # Listen for voice commands every 60 frames
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 60 == 0:
        listen_for_command()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
