"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io

import torch
from PIL import Image
from flask import Flask, request

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
    args = parser.parse_args()

    model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True)  # force_reload to recache
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
