# 👁️ YOLO-Lab

A hands-on laboratory for learning, experimenting, and mastering **real-time object detection and AI vision** using YOLOv8, OpenCV, MediaPipe, and Python.  
This repository takes you from basic object detection to advanced multimodal AI systems such as **voice-controlled detection** and **motion trail visualization**.

Whether you’re exploring YOLO for the first time or refining your AI skills, each section builds upon the previous — guiding you through detection, tracking, counting, and interactive applications step by step.  
The goal isn’t just to run models, but to understand **how AI perceives and interacts with the environment**.

> 💡 Perfect for students, researchers, and curious minds who love experimenting with visual intelligence.

---

## 📂 Repository Structure
```
yolo-lab/
│
├── README.md
├── requirements.txt
│
├── yolov8n-seg.pt
├── yolov8n.pt
│
├── 01_setup/
│ ├── 01_install_ultralytics.py # Install YOLOv8 and dependencies
│ ├── 02_check_gpu_status.py # Verify GPU availability
│ └── 03_load_pretrained_model.py # Load YOLOv8 pretrained weights
│
├── 02_training/
│ ├── 01_train_detection_coco8.py # Train YOLO on COCO8 detection dataset
│ ├── 02_train_segmentation_coco8.py # Train YOLO segmentation model
│ ├── 03_custom_dataset_training.py # Train on user-defined datasets
│ ├── 04_loss_visualization.ipynb # Visualize training losses
│ └── 05_model_evaluation.py # Evaluate trained models
│
├── 03_inference/
│ ├── 01_infer_images.py # Run detection on static images
│ ├── 02_infer_videos.py # Run detection on video files
│ ├── 03_infer_webcam.py # Run detection on live webcam feed
│ ├── 04_batch_processing.py # Process a batch of images/videos
│ └── 05_real_time_demo.py # General real-time demo script
│
├── 04_advanced/
│ ├── 01_yolo_pose_estimation.py # YOLO + pose estimation demo
│ ├── 02_yolo_segmentation_finetune.py # Fine-tune segmentation model
│ ├── 03_yolo_export_onnx_tflite.py # Export models to ONNX/TFLite
│ └── 04_yolo_custom_callbacks.py # Advanced training callbacks
│
├── 05_projects/
│ ├── 01_vehicle_detection.py # Detects vehicles in images/videos
│ ├── 02_people_counting.py # Counts people in real-time
│ ├── 03_realtime_surveillance.py # Lightweight surveillance demo
│ ├── 04_yolo_mediapipe_fusion.py # YOLO + MediaPipe keypoint detection
│ ├── 05_object_trail_tracker.py # Tracks an object and draws its trail
│ └── 06_voice_controlled_object_detection.py # Detect objects via voice commands
│
├── data/
│ ├── coco8.yaml # Detection dataset config
│ ├── coco8-seg.yaml # Segmentation dataset config
│ ├── sample_videos/ # Sample videos for testing
│ └── sample_images/ # Sample images for testing
│
├── runs/
│ ├── detect/ # Detection outputs
│ ├── segment/ # Segmentation outputs
│ └── pose/ # Pose estimation outputs
```
---

## 🧠 Learning Goals

- Understand **real-time object detection** with YOLOv8.
- Explore **tracking and motion visualization** (e.g., object trails).
- Count and monitor people or vehicles using live feeds.
- Fuse multiple AI systems (YOLO + MediaPipe keypoints, or voice commands).
- Learn **practical AI project workflows**: training, evaluation, inference, and deployment.
- Experiment with **multimodal inputs**: video, webcam, and speech.

---

## 📌 Projects Overview (05_projects/)

| File | Description |
|------|-------------|
| **01_vehicle_detection.py** | Detects vehicles in images/videos using YOLOv8. Draws bounding boxes and confidence scores. Ideal for traffic monitoring. |
| **02_people_counting.py** | Counts people in real-time. Demonstrates aggregation of detections over time. Useful for crowd monitoring or occupancy tracking. |
| **03_realtime_surveillance.py** | Lightweight real-time surveillance system. Detects moving objects and provides visual alerts. |
| **04_yolo_mediapipe_fusion.py** | Combines YOLO object detection with MediaPipe keypoint tracking (pose/hand/face). Great for gesture and pose-based applications. |
| **05_object_trail_tracker.py** | Tracks a target object (like a phone) and draws its motion trail dynamically across frames. |
| **06_voice_controlled_object_detection.py** | Detect objects by voice commands. Integrates speech recognition with YOLO detection for a multimodal AI experience. |

---

### 🔹 Notes

- All projects use **YOLOv8** lightweight models for **real-time performance**.
- Some demos require **MediaPipe** or **speech recognition** for advanced interactivity.
- Scripts serve as **foundations for experimentation**, extension, or integration into larger AI systems.
- The repository emphasizes **learning by doing**, exploring the full AI vision workflow from training to deployment.

---

## 🚀 Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/yolo-lab.git
cd yolo-lab

# Create a virtual environment
conda create -n yolo-env python=3.10
conda activate yolo-env

# Install dependencies
pip install -r requirements.txt
```
---
## 📬Feedback
For feedback or questions, contact: [barkin.adiguzel@gmail.com](mailto:barkin.adiguzel@gmail.com)












