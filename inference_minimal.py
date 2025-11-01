# inference_minimal.py
import cv2
import torch

# Load a pretrained YOLOv5 model from PyTorch Hub
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# Load image
img_path = r"yolov5\data\images\bus.jpg"  # Default image in YOLOv5 repo
img = cv2.imread(img_path)
assert img is not None, f"Image not found: {img_path}"

# Convert BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Inference
results = model(img)

# Print and show results
results.print()
results.show()
results.save(save_dir="output")
