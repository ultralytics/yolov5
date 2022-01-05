# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.mlmodel            # CoreML (under development)
                                         yolov5s_openvino_model     # OpenVINO (under development)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow protobuf
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                         yolov5s.engine             # TensorRT
"""

import os
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device


def run_model(weights, data, conf_thres, iou_thres, max_det, source, device = '', half = False, imgsz = [640,640]):
    print("[INFO] Computing face mask detections...")
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    for _, im, im0s, _, _ in dataset:
        faces, locs = detect_on_image(im, im0s, model, conf_thres, iou_thres, max_det, device, half)
    print("[INFO] Find", len(faces), "mask faces")
    return faces, locs

def find_points(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    return (int(x1 - w * 0.114), int(y1 - h * 0.434), int(x2 + w * 0.114), int(y2))

def detect_on_image(im, im0s, model, conf_thres, iou_thres, max_det, device, half):
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        faces = []
        locs = []

        # Process predictions
        for det in pred:  # per image
            im0 = im0s.copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, _, cls in reversed(det):
                    c = int(cls)
                    if (c == 0):
                        (startX, startY, endX, endY) = find_points(xyxy[0],xyxy[1],xyxy[2],xyxy[3])
                        faces.append(im0[startY:endY, startX:endX])
                        locs.append((startX, startY, endX, endY))
        return faces, locs
