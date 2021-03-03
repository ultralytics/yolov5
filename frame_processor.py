import torch
import numpy as np

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, xyxy2xywh, set_logging

from utils.torch_utils import select_device
from utils.datasets import letterbox


"""
Simplified version of original detect.py, where this class is initialized via an instance, then the user passes a frame
to the detect_cans method which returns a list of 4-tuples containing the detection locations.
"""


class TrainedModelFrameProcessor:
    def __init__(self, weights, img_size, conf_thresh, iou_thresh):
        self.weights = weights
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # Initialize
        set_logging()
        self.device = select_device('0')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = check_img_size(self.img_size, s=self.stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))  # run once

    def detect_cans(self, frame):

        """Taken from ./utils/datasets.py"""
        img = letterbox(frame, self.img_size, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        """"""

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh)

        # Process detections.

        detections = []  # Collects all the detections from one frame into a list of 4-tuples [(x1, y1, w1, h1), (x2, y2, w2, h2)]
        for i, det in enumerate(pred):  # detections per image
            s, im0 = '', frame

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # parse detections
                save_conf = False
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format

                    detection_ = line[1:]  # x,y,w,h
                    detections.append(detection_)  # Appending detections for one frame

        return detections
