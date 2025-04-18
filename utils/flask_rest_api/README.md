<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Flask REST API for YOLOv5

[Representational State Transfer (REST)](https://en.wikipedia.org/wiki/Representational_state_transfer) [Application Programming Interfaces (APIs)](https://en.wikipedia.org/wiki/API) provide a standardized way to expose [Machine Learning (ML)](https://www.ultralytics.com/glossary/machine-learning-ml) models for use by other services or applications. This directory contains an example REST API built with the [Flask](https://flask.palletsprojects.com/en/stable/) web framework to serve the [Ultralytics YOLOv5s](https://docs.ultralytics.com/models/yolov5/) model, loaded directly from [PyTorch Hub](https://pytorch.org/hub/ultralytics_yolov5/). This setup allows you to easily integrate YOLOv5 [object detection](https://docs.ultralytics.com/tasks/detect/) capabilities into your web applications or microservices, aligning with common [model deployment options](https://docs.ultralytics.com/guides/model-deployment-options/).

## 💻 Requirements

The primary requirement is the [Flask](https://flask.palletsprojects.com/en/stable/) web framework. You can install it using pip:

```shell
pip install Flask
```

You will also need `torch` and `yolov5`. These are implicitly handled by the script when it loads the model from PyTorch Hub. Ensure you have a functioning Python environment set up.

## ▶️ Run the API

Once Flask is installed, you can start the API server using the following command:

```shell
python restapi.py --port 5000
```

The server will begin listening on the specified port (defaulting to 5000). You can then send inference requests to the API endpoint using tools like [curl](https://curl.se/) or any other HTTP client.

To test the API with a local image file (e.g., `zidane.jpg` located in the `yolov5/data/images` directory relative to the script):

```shell
curl -X POST -F image=@../data/images/zidane.jpg 'http://localhost:5000/v1/object-detection/yolov5s'
```

The API processes the submitted image using the YOLOv5s model and returns the detection results in [JSON](https://www.json.org/json-en.html) format. Each object within the JSON array represents a detected item, including its class ID, confidence score, normalized [bounding box](https://www.ultralytics.com/glossary/bounding-box) coordinates (`xcenter`, `ycenter`, `width`, `height`), and class name.

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

An example Python script, `example_request.py`, is included to demonstrate how to perform inference using the popular [requests](https://requests.readthedocs.io/en/latest/) library. This script offers a straightforward method for interacting with the running API programmatically.

## 🤝 Contribute

Contributions to enhance this Flask API example are highly encouraged! Whether you're interested in adding support for different YOLO models, improving error handling, or implementing new features, please feel free to fork the repository, apply your changes, and submit a pull request. For more comprehensive contribution guidelines, please refer to the main [Ultralytics YOLOv5 repository](https://github.com/ultralytics/yolov5) and the general [Ultralytics documentation](https://docs.ultralytics.com/).
