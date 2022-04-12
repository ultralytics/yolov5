# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run a Flask REST API exposing a YOLOv5s model
"""

import argparse
import io

import torch
from flask import Flask, request
from PIL import Image

app = Flask(__name__)

DETECTION_URL = "/v1/object-detection/yolov5s"


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()

        img = Image.open(io.BytesIO(image_bytes))

        results = model(img, size=640)  # reduce size=320 for faster inference
        return results.pandas().xyxy[0].to_json(orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    opt = parser.parse_args()

    # Fix known issue urllib.error.HTTPError 403: rate limit exceeded https://github.com/ultralytics/yolov5/pull/7210
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

    model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True)  # force_reload to recache
    app.run(host="0.0.0.0", port=opt.port)  # debug=True causes Restarting with stat
