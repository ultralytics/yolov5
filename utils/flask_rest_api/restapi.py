# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run a Flask REST API exposing one or more YOLOv5s models."""

import argparse
import io

import torch
from flask import Flask, request
from PIL import Image

app = Flask(__name__)
models = {}

DETECTION_URL = "/v1/object-detection/<model>"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"}
MAX_IMAGE_SIZE = 16 * 1024 * 1024  # 16 MB


@app.route(DETECTION_URL, methods=["POST"])
def predict(model):
    """Predict and return object detections in JSON format given an image and model name via a Flask REST API POST
    request.
    """
    if request.method != "POST":
        return

    if request.files.get("image"):
        im_file = request.files["image"]

        # Validate file extension against allowlist
        filename = im_file.filename or ""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            return {"error": "Invalid file type. Allowed types: " + ", ".join(ALLOWED_EXTENSIONS)}, 400

        # Enforce upload size limit
        im_bytes = im_file.read(MAX_IMAGE_SIZE + 1)
        if len(im_bytes) > MAX_IMAGE_SIZE:
            return {"error": "File too large. Maximum size is 16 MB."}, 413

        im = Image.open(io.BytesIO(im_bytes))
        im.verify()  # Verify it is a valid image (raises exception if not)
        im = Image.open(io.BytesIO(im_bytes))  # Re-open after verify (verify closes the stream)

        if model in models:
            results = models[model](im, size=640)  # reduce size=320 for faster inference
            return results.pandas().xyxy[0].to_json(orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument("--model", nargs="+", default=["yolov5s"], help="model(s) to run, i.e. --model yolov5n yolov5s")
    opt = parser.parse_args()

    for m in opt.model:
        models[m] = torch.hub.load("ultralytics/yolov5", m, force_reload=True, skip_validation=True)

    app.run(host="127.0.0.1", port=opt.port)  # debug=True causes Restarting with stat
