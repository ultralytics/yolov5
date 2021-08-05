# Flask REST API
[REST](https://en.wikipedia.org/wiki/Representational_state_transfer) [API](https://en.wikipedia.org/wiki/API)s are commonly used to expose Machine Learning (ML)  models to other services. This folder contains an example REST API created using Flask to expose the YOLOv5s model from [PyTorch Hub](https://pytorch.org/hub/ultralytics_yolov5/).

## Requirements

[Flask](https://palletsprojects.com/p/flask/) is required. Install with:
```shell
$ pip install Flask
```

## Run

After Flask installation run:

```shell
$ python3 restapi.py --port 5000
```

Then use [curl](https://curl.se/) to perform a request:

```shell
$ curl -X POST -F image=@zidane.jpg 'http://localhost:5000/v1/object-detection/yolov5s'
```

The model inference results are returned as a JSON response:

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

An example python script to perform inference using [requests](https://docs.python-requests.org/en/master/) is given in `example_request.py`
