import os

import gradio as gr
import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


def yolo(img):
    results = model(img)  # inference
    results.render()  # updates results.imgs with boxes and labels
    return Image.fromarray(results.imgs[0])  # return annotated PIL Image


inputs = gr.inputs.Image(type='pil', label="Original Image")
outputs = gr.outputs.Image(type="pil", label="Output Image")

title = "YOLOv5"
description = "YOLOv5s demo for object detection. Upload an image or click an example image to use. " \
              "Source code: https://github.com/ultralytics/yolov5"
article = "YOLOv5 is a family of compound-scaled object detection models trained on the COCO dataset, and includes " \
          "simple functionality for Test Time Augmentation (TTA), model ensembling, hyperparameter evolution, " \
          "and export to ONNX, CoreML and TFLite. Source code: https://github.com/ultralytics/yolov5, " \
          "iOS App: https://apps.apple.com/app/id1452689527, PyTorch Hub: https://pytorch.org/hub/ultralytics_yolov5"

examples = [
    ['bird.jpg'],
    ['fox.jpg']
]

gr.Interface(yolo, inputs, outputs, title=title, description=description, article=article, examples=examples).launch(
    debug=True)
