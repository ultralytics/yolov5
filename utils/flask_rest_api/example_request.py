# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Perform test request."""

import os
import pprint

import requests

DETECTION_URL = "http://localhost:5000/v1/object-detection/yolov5s"
IMAGE = "../../data/images/zidane.jpg"

# Read image
with open(IMAGE, "rb") as f:
    image_data = f.read()

# Optional: set API_KEY env var to authenticate (e.g. API_KEY=mysecretkey python example_request.py)
headers = {}
if api_key := os.environ.get("API_KEY", ""):
    headers["X-API-Key"] = api_key

# Send the filename so the server can validate the extension (restapi.py checks ALLOWED_EXTENSIONS)
response = requests.post(
    DETECTION_URL,
    files={"image": ("zidane.jpg", image_data, "image/jpeg")},
    headers=headers,
).json()

pprint.pprint(response)
