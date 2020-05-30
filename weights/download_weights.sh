#!/bin/bash
# Download common models

python3 -c "from models import *;
attempt_download('weights/yolov5s.pt');
attempt_download('weights/yolov5m.pt');
attempt_download('weights/yolov5l.pt')"
