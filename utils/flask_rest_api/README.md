<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Flask REST API for YOLOv5

[Representational State Transfer (REST)](https://en.wikipedia.org/wiki/Representational_state_transfer) [Application Programming Interfaces (APIs)](https://en.wikipedia.org/wiki/API) are a standard way to expose [Machine Learning (ML)](https://www.ultralytics.com/glossary/machine-learning-ml) models for consumption by other services or applications. This directory provides an example REST API built with [Flask](https://flask.palletsprojects.com/en/stable/) to serve the Ultralytics [YOLOv5s](https://docs.ultralytics.com/models/yolov5/) model loaded directly from [PyTorch Hub](https://pytorch.org/hub/ultralytics_yolov5/). This allows you to easily integrate YOLOv5 object detection capabilities into your web applications or microservices.

## 💻 Requirements

The primary requirement is the [Flask](https://flask.palletsprojects.com/en/stable/) web framework. Install it using pip:

```shell
pip install Flask
```

You will also need `torch` and `yolov5` which are implicitly handled when loading the model from PyTorch Hub in the script. Ensure you have a working Python environment.

## ▶️ Run the API

Once Flask is installed, you can start the API server:

```shell
python restapi.py --port 5000
```

The server will start listening on the specified port (default is 5000). You can then send inference requests to the API endpoint using tools like [curl](https://curl.se/) or any HTTP client.

To test the API with an image file (e.g., `zidane.jpg` from the `yolov5/data/images` directory):

```shell
curl -X POST -F image=@../data/images/zidane.jpg 'http://localhost:5000/v1/object-detection/yolov5s'
```

The API processes the image using the YOLOv5s model and returns the detection results in [JSON](https://www.json.org/json-en.html) format. Each object in the JSON array represents a detected object with its class ID, confidence score, bounding box coordinates (normalized), and class name.

```json
[
  {
    "class": 0,
    "confidence": 0.8900438547,
    "height": 0.9318675399,
    "name": "person",
    "width": 0.3264600933,
    "xcenter": 0.7438579798,
    "ycenter": 0.5207948685
  },
  {
    "class": 0,
    "confidence": 0.8440024257,
    "height": 0.7155083418,
    "name": "person",
    "width": 0.6546785235,
    "xcenter": 0.427829951,
    "ycenter": 0.6334488392
  },
  {
    "class": 27,
    "confidence": 0.3771208823,
    "height": 0.3902671337,
    "name": "tie",
    "width": 0.0696444362,
    "xcenter": 0.3675483763,
    "ycenter": 0.7991207838
  },
  {
    "class": 27,
    "confidence": 0.3527112305,
    "height": 0.1540903747,
    "name": "tie",
    "width": 0.0336618312,
    "xcenter": 0.7814827561,
    "ycenter": 0.5065554976
  }
]
```

An example Python script, `example_request.py`, demonstrates how to perform inference using the popular [requests](https://requests.readthedocs.io/en/latest/) library. This script provides a simple way to interact with the running API programmatically.

## 🤝 Contribute

Contributions to enhance this Flask API example are welcome! Whether it's adding support for different YOLO models, improving error handling, or adding new features, feel free to fork the repository, make your changes, and submit a pull request. Check out the main [Ultralytics YOLOv5 repository](https://github.com/ultralytics/yolov5) for more details on contributing.
