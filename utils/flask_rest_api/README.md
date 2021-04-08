## Flask Rest API
Rest API are commonly used to expose machine learning models to other services. This folder contains an example rest API created using Flask to expose the `yolov5s` model from pytorch hub.

Install Flask in your environment and run:

`$ python3 restapi.py --port 5000`

Then use [curl](https://curl.se/) to perform a request:

`$ curl -X POST -F image=@zidane.jpg 'http://localhost:5000/v1/object-detection/yolov5s'`

The model inference results are returned:

```
[{'class': 0,
  'confidence': 0.8197850585,
  'name': 'person',
  'xmax': 1159.1403808594,
  'xmin': 750.912902832,
  'ymax': 711.2583007812,
  'ymin': 44.0350036621},
 {'class': 0,
  'confidence': 0.5667674541,
  'name': 'person',
  'xmax': 1065.5523681641,
  'xmin': 116.0448303223,
  'ymax': 713.8904418945,
  'ymin': 198.4603881836},
 {'class': 27,
  'confidence': 0.5661227107,
  'name': 'tie',
  'xmax': 516.7975463867,
  'xmin': 416.6880187988,
  'ymax': 717.0524902344,
  'ymin': 429.2020568848}]
```

An example python script to perform inference using [requests](https://docs.python-requests.org/en/master/) is given in `example_request.py`
