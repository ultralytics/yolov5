## gratheon/models-bee-detector
Microservice that detects bees

- Essentially Uses Ultralytics yolov5 model
- Runs as a http server
- Dockerized
- Uses weights by Matt Nudi
https://github.com/mattnudi/bee-detection
https://universe.roboflow.com/matt-nudi/honey-bee-detection-model-zgjnb

## Usage
```
# webcam
python detect.py --weights yolov5s.pt --source 0

# video file
python detect.py --weights yolov5s.pt --source file.mp4
```


## Ultralytics yolo v5 license
YOLOv5 is available under two different licenses:

- **GPL-3.0 License**: See [LICENSE](https://github.com/ultralytics/yolov5/blob/master/LICENSE) file for details.
- **Enterprise License**: Provides greater flexibility for commercial product development without the open-source requirements of GPL-3.0. Typical use cases are embedding Ultralytics software and AI models in commercial products and applications. Request an Enterprise License at [Ultralytics Licensing](https://ultralytics.com/license).
