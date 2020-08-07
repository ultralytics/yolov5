import torch

link = 'https://github.com/ultralytics/yolov5/releases/download/v2.0/yolov5s.pt'
link = 'https://github.com/ultralytics/yolov5/releases/download/v2.0/yolov5m.pt'
name = 'yolov5s.pt'

torch.hub.download_url_to_file(link, name)