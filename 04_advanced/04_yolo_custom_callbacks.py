"""
Demonstration of using custom callbacks in YOLOv8 training.

This script:
1. Defines custom callback functions that trigger during training events
2. Registers them in the YOLO training pipeline
3. Prints useful logs such as training loss and final summary

Callbacks can be used for:
- Custom logging
- Saving metrics externally
- Integrating with dashboards or live monitoring tools
"""

from ultralytics import YOLO

# Define custom callback functions
def on_train_epoch_end(trainer):
    """Triggered at the end of each training epoch."""
    epoch = trainer.epoch + 1
    loss = trainer.loss_items
    print(f"[Callback] Epoch {epoch} finished - Loss: {loss}")

def on_train_end(trainer):
    """Triggered after training completes."""
    print("[Callback] Training complete! Model saved at:", trainer.best)

# Create a YOLO model (you can change this path to your own .pt file)
model = YOLO("yolov8n.pt")

# Register callbacks
model.add_callback("on_train_epoch_end", on_train_epoch_end)
model.add_callback("on_train_end", on_train_end)

# Start training with the custom callbacks
model.train(
    data="coco8.yaml",   # Example dataset
    epochs=5,            # Short training for demo
    imgsz=640,           # Image size
    batch=4,             # Batch size
    device="cpu",        # Use GPU if available: 'cuda'
    project="runs/callback_demo",
    name="yolo_callbacks"
)
