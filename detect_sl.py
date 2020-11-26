import argparse
import os
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
import supervisely_lib as sly
from supervisely_lib.annotation.tag_meta import TagValueType
import numpy as np



def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



def detect(save_img=False):
    weights = ['best.pt']

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    model = model.half() #@TODO to run change: return F.hardswish(input, self.inplace)' replace with 'return F.hardswish(input) in ~/.local/lib/python3.8/site-packages/torch/nn/modules/activation.py

    # Set Dataloader
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names  # ['lemon', 'kiwi']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    #create meta
    tag_meta = sly.TagMeta('confidence', TagValueType.ANY_NUMBER)
    obj_classes = []
    for class_name, class_color in zip(names, colors):
        obj_classes.append(sly.ObjClass(class_name, sly.Rectangle, class_color))

    new_collection = sly.ObjClassCollection(obj_classes)
    new_meta = sly.ProjectMeta(new_collection, sly.TagMetaCollection([tag_meta]))
    api.project.update_meta(new_project.id, new_meta.to_json())

    for dataset in datasets:
        new_dataset = api.dataset.create(new_project.id, dataset.name)
        images = api.image.get_list(dataset.id)
        for batch in sly.batched(images):
            image_ids = [image_info.id for image_info in batch]
            image_names = [image_info.name for image_info in batch]
            ann_infos = api.annotation.download_batch(dataset.id, image_ids)
            images_np = api.image.download_nps(dataset.id, image_ids)
            images_np_0 = [image[...,::-1] for image in images_np]
            images_np = []

            for img0 in images_np_0:
                img = letterbox(img0)[0]
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                images_np.append(np.ascontiguousarray(img))

            # Run inference
            res_anns = []
            for img, im0s, image_name, ann_info in zip(images_np, images_np_0, image_names, ann_infos):
                ann = sly.Annotation.from_json(ann_info.annotation, meta)
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                pred = model(img, augment=opt.augment)[0]
                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            top, left , bottom, right = int(xyxy[1]), int(xyxy[0]), int(xyxy[3]), int(xyxy[2])
                            new_geom = sly.Rectangle(top, left , bottom, right)
                            new_obj_class = new_collection.get(names[int(cls)])
                            tag = sly.Tag(tag_meta, round(float(conf), 2)) #tensor.item()
                            new_tag = sly.TagCollection([tag])
                            new_label = sly.Label(new_geom, new_obj_class, new_tag)
                            ann = ann.add_label(new_label)

                    res_anns.append(ann)

            new_image_infos = api.image.upload_ids(new_dataset.id, image_names, image_ids)
            image_ids = [img_info.id for img_info in new_image_infos]
            api.annotation.upload_anns(image_ids, res_anns)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    #parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='output confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    #print(opt)

    api = sly.Api.from_env()
    workspace_id = 23821
    project_id = 103137

    new_project = api.project.create(workspace_id, 'yolo5_check_lemon_kiwi', change_name_if_conflict=True)

    datasets = api.dataset.get_list(project_id)
    meta_json = api.project.get_meta(project_id)
    meta = sly.ProjectMeta.from_json(meta_json)

    with torch.no_grad():
            detect()
