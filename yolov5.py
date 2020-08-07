import os
import torch
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, plot_one_box
from utils.torch_utils import select_device


class Yolov5():
    def __init__(self, weights_path, device='', img_size=640, conf_thres=0.4, iou_thres=0.5, augment=False, agnostic_nms=False, classes=None):
        self.device = select_device(device)
        self.weights_name = os.path.split(weights_path)[-1]
        self.model = attempt_load(weights_path, map_location=self.device)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.imgsz = check_img_size(img_size, s=self.model.stride.max())
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.augment = augment
        self.agnostic_nms = agnostic_nms
        self.classes = classes
        self.half = self.device.type != 'cpu'
        if self.half:
            self.model.half()
        if self.device.type != 'cpu':
            self.burn()


    def __str__(self):
        out = ['Model: %s' % self.weights_name]
        out.append('Image size: %s' % self.imgsz)
        out.append('Confidence threshold: %s' % self.conf_thres)
        out.append('IoU threshold: %s' % self.iou_thres)
        out.append('Augment: %s' % self.augment)
        out.append('Agnostic nms: %s' % self.agnostic_nms)
        if self.classes != None:
            filter_classes = [self.names[each_class] for each_class in self.classes]
            out.append('Classes filter: %s' % filter_classes)
        out.append('Classes: %s' % self.names)
        
        return '\n'.join(out)


    def burn(self):
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img)  # run once


    def predict(self, img0, draw_bndbox=False, bndbox_format='min_max_list'):
        img = letterbox(img0, new_shape=self.imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)  # uint8 to float32  
        
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0 # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=self.augment)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

        det = pred[0]

        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        if draw_bndbox:
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (self.names[int(cls)], conf)
                plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)])

        if bndbox_format == 'min_max_list':
            min_max_list = self.min_max_list(det)
            return min_max_list


    def predict_batch(self):
        #TODO
        pass


    def min_max_list(self, det):
        min_max_list = []
        for i, c in enumerate(det[:, -1]):
            obj = {
                'bndbox': {
                    'xmin': min(int(det[i][0]),int(det[i][2])),
                    'xmax': max(int(det[i][0]),int(det[i][2])),
                    'ymin': min(int(det[i][1]),int(det[i][3])),
                    'ymax': max(int(det[i][1]),int(det[i][3]))
                    },
                'name': self.names[int(c)],
                'conf': float(det[i][4]),
                'color': self.colors[int(det[i][5])]
                }
            min_max_list.append(obj)

        return min_max_list