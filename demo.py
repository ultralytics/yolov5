import torch
from PIL import Image
from torchvision import transforms
import gradio as gr
from io import BytesIO
import base64
import os


# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def yolo(img):
  img.save("test_image_hack.png")
  img = Image.open("test_image_hack.png")
  results = model(img)  # inference
  os.remove("test_image_hack.png")

  results.imgs # array of original images (as np array) passed to model for inference
  results.render()  # updates results.imgs with boxes and labels
  for img in results.imgs:
    return Image.fromarray(img)


inputs =  gr.inputs.Image(type='pil', label="Original Image")
outputs = gr.outputs.Image(type="pil",label="Output Image")

title = "YOLOv5"
description = "demo for YOLOv5 which takes in a single image for object detection. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<a href='https://github.com/ultralytics/yolov5'>Github Repo</a></p>"

examples = [
    ['bird.jpg'],
    ['fox.jpg']
]
gr.Interface(yolo, inputs, outputs, title=title, description=description, article=article, examples=examples).launch(debug=True)