# This script checks if your environment is correctly set up for YOLOv8.
# It verifies CUDA (GPU) availability, and prints the installed versions of PyTorch, OpenCV, and Ultralytics.

import torch
import cv2

def check_gpu_status():
    print("Checking GPU and library setup...\n")

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available. GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is NOT available. Running on CPU only.")

    # Print library versions
    print(f"OpenCV version: {cv2.__version__}")
    print(f"PyTorch version: {torch.__version__}")

    try:
        import ultralytics
        print(f"Ultralytics (YOLO) version: {ultralytics.__version__}")
    except ImportError:
        print("Ultralytics (YOLO) is not installed. Run: pip install ultralytics")

    print("\nSystem check complete.\n")

if __name__ == "__main__":
    check_gpu_status()
