<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Flask REST API for YOLOv5

[Representational State Transfer (REST)](https://en.wikipedia.org/wiki/Representational_state_transfer) [Application Programming Interfaces (APIs)](https://en.wikipedia.org/wiki/API) provide a standardized way to expose [Machine Learning (ML)](https://www.ultralytics.com/glossary/machine-learning-ml) models for use by other services or applications. This directory contains an example REST API built with the [Flask](https://flask.palletsprojects.com/en/stable/) web framework to serve the [Ultralytics YOLOv5s](https://docs.ultralytics.com/models/yolov5) model, loaded directly from [PyTorch Hub](https://pytorch.org/hub/ultralytics_yolov5/). This setup allows you to easily integrate YOLOv5 [object detection](https://docs.ultralytics.com/tasks/detect) capabilities into your web applications or microservices, aligning with common [model deployment options](https://docs.ultralytics.com/guides/model-deployment-options).

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

To test the API with a local image file (e.g., `zidane.jpg` located in the `yolov5/data/images` directory, which is `../../data/images/zidane.jpg` relative to the script):

```shell
curl -X POST -F image=@../../data/images/zidane.jpg 'http://localhost:5000/v1/object-detection/yolov5s'
```

The API processes the submitted image using the YOLOv5s model and returns the detection results in [JSON](https://www.json.org/json-en.html) format. Each object within the JSON array represents a detected item, including its pixel [bounding box](https://www.ultralytics.com/glossary/bounding-box) coordinates (`xmin`, `ymin`, `xmax`, `ymax`), confidence score, class ID, and class name.

```json
[
  {
    "xmin": 749.5,
    "ymin": 43.5,
    "xmax": 1148.0,
    "ymax": 704.5,
    "confidence": 0.8900438547,
    "class": 0,
    "name": "person"
  },
  {
    "xmin": 113.5,
    "ymin": 196.0,
    "xmax": 1093.0,
    "ymax": 711.0,
    "confidence": 0.8440024257,
    "class": 0,
    "name": "person"
  },
  {
    "xmin": 437.5,
    "ymin": 433.5,
    "xmax": 529.5,
    "ymax": 717.5,
    "confidence": 0.3771208823,
    "class": 27,
    "name": "tie"
  },
  {
    "xmin": 1090.0,
    "ymin": 312.0,
    "xmax": 1135.0,
    "ymax": 410.0,
    "confidence": 0.3527112305,
    "class": 27,
    "name": "tie"
  }
]
```

An example Python script, `example_request.py`, is included to demonstrate how to perform inference using the popular [requests](https://requests.readthedocs.io/en/latest/) library. This script offers a straightforward method for interacting with the running API programmatically.

## 🤝 Contribute

Contributions to enhance this Flask API example are highly encouraged! Whether you're interested in adding support for different YOLO models, improving error handling, or implementing new features, please feel free to fork the repository, apply your changes, and submit a pull request. For more comprehensive contribution guidelines, please refer to the main [Ultralytics YOLOv5 repository](https://github.com/ultralytics/yolov5) and the general [Ultralytics documentation](https://docs.ultralytics.com/).
