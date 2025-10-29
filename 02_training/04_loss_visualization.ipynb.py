"""
This notebook visualizes YOLO training metrics (loss curves, mAP, precision, recall)
to help understand how the model’s performance evolved over epochs.

NOTE:
This assumes you've already trained a YOLO model and have a `results.csv` file.
By default, YOLO saves this file in `runs/detect/<experiment_name>/results.csv`.
If you trained a custom model, update the `results_path` variable below.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================
# 1. Load YOLO Training Logs
# ==============================
# Change this path according to your experiment folder
results_path = 'runs/detect/yolov8_detection/results.csv'

if not os.path.exists(results_path):
    raise FileNotFoundError(f"Could not find results file at: {results_path}\nMake sure you trained a model first.")

# Load training results into pandas DataFrame
df = pd.read_csv(results_path)

print("Columns in results.csv:", df.columns.tolist())

# ==============================
# 2. Plot Loss Curves
# ==============================
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss')
plt.plot(df['epoch'], df['train/cls_loss'], label='Class Loss')
plt.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# ==============================
# 3. Plot Performance Metrics (mAP, Precision, Recall)
# ==============================
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Model Performance Metrics')
plt.legend()
plt.grid(True)
plt.show()

print("Loss and metric visualization complete.")
