import os
import torch
import numpy as np

import json
import xml.etree.cElementTree as ET
from lxml import etree

from utils.datasets import LoadImages
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

        self.annotation_writer = AnnotationWriter()


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


    def predict(self, img0, img=None, draw_bndbox=False, bndbox_format='min_max_list'):
        if img is None:
            img = self.send_whatever_to_device(img0)
        else:
            img = self.send_to_device(img)

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


    def predict_batch(self, img0s, draw_bndbox=False, bndbox_format='min_max_list'):

        imgs = self.send_whatever_to_device(img0s)

        with torch.no_grad():
            # Run model
            inf_out, _ = self.model(imgs, augment=self.augment)  # inference and training outputs

            # Run NMS
            preds = non_max_suppression(inf_out, conf_thres=self.conf_thres, iou_thres=self.iou_thres)

        batch_output = []
        for pred in preds:
            batch_output.append(self.min_max_list(pred))

        return batch_output


    def predict_from_path_to_path(self, from_path='inference/images/', to_path='output/', draw_bnd_box=False, bndbox_format='labelimg'):
        dataset = LoadImages(from_path, img_size=self.imgsz)

        for path, img, im0s, vid_cap in dataset:
            if bndbox_format == 'yolo':
                #TODO
                pass
            elif bndbox_format == 'labelimg':
                detections = self.predict(im0s, img, draw_bndbox=draw_bnd_box, bndbox_format='min_max_list')
                img_name = os.path.split(path)[-1]
                self.annotation_writer.write_labelimg(detections, im0s.shape, img_name, to_path)
            elif bndbox_format == 'udt':
                #TODO
                pass
            elif bndbox_format == 'aws':
                #TODO
                pass


    def send_to_device(self, img_to_send):
        img_to_send = torch.from_numpy(img_to_send).to(self.device)
        img_to_send = img_to_send.half() if self.half else img_to_send.float()
        img_to_send /= 255.0 # 0 - 255 to 0.0 - 1.0
        if img_to_send.ndimension() == 3:
            img_to_send = img_to_send.unsqueeze(0)

        return img_to_send

    
    def reshape_copy_img(self, img):
        _img = letterbox(img, new_shape=self.imgsz)[0]
        _img = _img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        _img = np.ascontiguousarray(_img)  # uint8 to float32
        return _img


    def send_whatever_to_device(self, img_s):
        if isinstance(img_s, list):
            img_to_send = []
            for img in img_s:
                img_to_send.append(self.reshape_copy_img(img))
            img_to_send = np.array(img_to_send)
        elif isinstance(img_s, np.ndarray):
            img_to_send = self.reshape_copy_img(img_s)
        else:
            print(type(img_s), ' is not supported')
            raise

        img_to_send = self.send_to_device(img_to_send)

        return img_to_send


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


class AnnotationWriter():
    def __init__(self):
        pass

    def write_udt(self):
        pass

    def write_labelimg(self, detections, image_shape, img_name, savedir):
        self.write_xml(detections, image_shape, img_name, savedir)

    def write_xml(self, detections, img_shape, img_name, savedir):
        xml_name = img_name.replace('.jpg', '.xml')
        if not os.path.isdir(savedir):
            os.mkdir(savedir)

        height, width, depth = img_shape

        annotation = ET.Element('annotation')
        ET.SubElement(annotation, 'filename').text = img_name
        size = ET.SubElement(annotation, 'size')
        ET.SubElement(size, 'width').text = str(width)
        ET.SubElement(size, 'height').text = str(height)
        ET.SubElement(size, 'depth').text = str(depth)
        for obj in detections:
            label = obj['name']

            ob = ET.SubElement(annotation, 'object')
            ET.SubElement(ob, 'conf').text = str(obj["conf"])
            ET.SubElement(ob, 'name').text = label
            bbox = ET.SubElement(ob, 'bndbox')
            ET.SubElement(bbox, 'xmin').text = str(int(obj['bndbox']['xmin']))
            ET.SubElement(bbox, 'ymin').text = str(int(obj['bndbox']['ymin']))
            ET.SubElement(bbox, 'xmax').text = str(int(obj['bndbox']['xmax']))
            ET.SubElement(bbox, 'ymax').text = str(int(obj['bndbox']['ymax']))

        xml_str = ET.tostring(annotation)
        root = etree.fromstring(xml_str)
        xml_str = etree.tostring(root, pretty_print=True)

        save_path = os.path.join(savedir, xml_name)
        with open(save_path, 'wb') as _writer:
            _writer.write(xml_str)

        return xml_str