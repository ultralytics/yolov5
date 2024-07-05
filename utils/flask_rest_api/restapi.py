# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Run a Flask REST API exposing one or more YOLOv5s models."""

import argparse
import io

import torch
from flask import Flask, request
from PIL import Image

app = Flask(__name__)
models = {}

DETECTION_URL = "/v1/object-detection/<model>"


@app.route(DETECTION_URL, methods=["POST"])
def predict(model):
    """
    Predict and return object detections in JSON format given an image and model name via a Flask REST API POST request.

    Args:
        model (str): The name of the YOLOv5 model to be used for prediction. This should be one of the models loaded into
            the `models` dictionary.

    Returns:
        (dict | None): JSON-formatted dictionary containing predicted object detections if the POST request is valid and
            contains an image file; otherwise, returns None.

    Notes:
        - Ensure the Flask application is properly configured and running to handle POST requests at the endpoint
          `DETECTION_URL`.
        - The image file should be included in the POST request with the key "image".
        - The model specified in the request should already be loaded into the `models` dictionary.

    Example:
    ```python
    import requests

    url = "http://localhost:5000/v1/object-detection/yolov5s"
    files = {"image": open("image.jpg", "rb")}
    response = requests.post(url, files=files)

    print(response.json())
    ```
    """
    if request.method != "POST":
        return

    if request.files.get("image"):
        # Method 1
        # with request.files["image"] as f:
        #     im = Image.open(io.BytesIO(f.read()))

        # Method 2
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

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

    app.run(host="0.0.0.0", port=opt.port)  # debug=True causes Restarting with stat
