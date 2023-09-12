"""
Run YOLOv5 detection inference on Gradio

Usage:
    pip install gradio
    python run.py
"""
import gradio as gr
import torch

# Load the local YOLOv5 model
model = torch.hub.load("./", "custom", path="yolov5s.pt", source="local")

# Set the interface title and description
title = "Ultralytics YOLOv5"
desc = "Default Model: yolov5s"

# Set default confidence and IoU thresholds
base_conf, base_iou = 0.25, 0.45


# Define a function for detecting objects in an image
def det_image(img, conf_thres, iou_thres):
    # Set the model's confidence and IoU thresholds
    model.conf = conf_thres
    model.iou = iou_thres
    # Run the model and return the first image from the detection results
    return model(img).render()[0]


# Create a Gradio interface
gr.Interface(
    # Specify input components as an image and two sliders to adjust confidence and IoU thresholds
    inputs=["image", gr.Slider(minimum=0, maximum=1, value=base_conf),
            gr.Slider(minimum=0, maximum=1, value=base_iou)],
    # Output component is an image
    outputs=["image"],
    # Specify the function to call
    fn=det_image,
    # Set the title and description of the interface
    title=title,
    description=desc,
    # Enable live preview
    live=True,
    # Provide some example inputs
    examples=[["data/images/bus.jpg", base_conf, base_iou],
              ["data/images/zidane.jpg", 0.3, base_iou]]
).launch(share=False)  # Launch the Gradio interface and disable sharing option
