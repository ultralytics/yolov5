import os

import gradio as gr
import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)


def yolo(img):
    basewidth = 512
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save("test_image_hack.png")
    img = Image.open("test_image_hack.png")
    results = model(img)  # inference
    os.remove("test_image_hack.png")

    results.render()  # updates results.imgs with boxes and labels
    for img in results.imgs:
        return Image.fromarray(img)


inputs = gr.inputs.Image(type='pil', label="Original Image")
outputs = gr.outputs.Image(type="pil", label="Output Image")

title = "YOLOv5"
description = "YOLOv5 demo for object detection. Upload an image or click an example image to use."
article = "<p style='text-align: center'>YOLOv5 is a family of compound-scaled object detection models trained on the COCO dataset, and includes " \
          "simple functionality for Test Time Augmentation (TTA), model ensembling, hyperparameter evolution, " \
          "and export to ONNX, CoreML and TFLite. <a href='https://github.com/ultralytics/yolov5'>Source code</a> |" \
          "<a href='https://apps.apple.com/app/id1452689527'>iOS App</a> | <a href='https://pytorch.org/hub/ultralytics_yolov5'>PyTorch Hub</a></p>"

examples = [
    ['https://user-images.githubusercontent.com/81195143/112767464-f0c9bb00-8fe4-11eb-9df0-e6edef249294.jpg'],
    ['https://user-images.githubusercontent.com/81195143/112767474-f7f0c900-8fe4-11eb-9581-6d9dab42b126.jpg']
]
gr.Interface(yolo, inputs, outputs, title=title, description=description, article=article, examples=examples).launch(
    debug=True)
