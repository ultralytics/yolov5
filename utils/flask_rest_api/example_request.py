"""Perform test request"""
import pprint

import requests

DETECTION_URL = "http://localhost:5000/v1/object-detection/yolov5s"
TEST_IMAGE = "zidane.jpg"

image_data = open(TEST_IMAGE, "rb").read()

response = requests.post(DETECTION_URL, files={"image": image_data}).json()

pprint.pprint(response)
