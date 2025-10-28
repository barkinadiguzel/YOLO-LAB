# 01_install_ultralytics.py
# Code to install the Ultralytics YOLO package and required dependencies

import os
import sys
import subprocess

def install_ultralytics():
    print("Installing Ultralytics YOLO package...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "ultralytics"])
        print("Ultralytics installed successfully.")
    except subprocess.CalledProcessError:
        print("Installation failed. Please check your internet connection or Python environment.")

def verify_installation():
    """Verifies that YOLO can be imported and version is shown."""
    try:
        import ultralytics
        print(f"YOLO version: {ultralytics.__version__}")
    except ImportError:
        print("YOLO is not installed properly.")

if __name__ == "__main__":
    install_ultralytics()
    verify_installation()
