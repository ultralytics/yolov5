import torch
import numpy as np
import supervisely as sly
from supervisely.io.fs import get_file_name_with_ext
import os
from pathlib import Path
import yaml
import torch.nn.functional as F
import cv2
import itertools

from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xywh2xyxy
from utils.datasets import letterbox
from supervisely.geometry.sliding_windows_fuzzy import SlidingWindowsFuzzy, SlidingWindowBorderStrategy


CONFIDENCE = "confidence"
IMG_SIZE = 640


def construct_model_meta(model):
    names = model.module.names if hasattr(model, 'module') else model.names

    colors = None
    if hasattr(model, 'module') and hasattr(model.module, 'colors'):
        colors = model.module.colors
    elif hasattr(model, 'colors'):
        colors = model.colors
    else:
        colors = []
        for i in range(len(names)):
            colors.append(sly.color.generate_rgb(exist_colors=colors))

    obj_classes = [sly.ObjClass(name, sly.Rectangle, color) for name, color in zip(names, colors)]
    tags = [sly.TagMeta(CONFIDENCE, sly.TagValueType.ANY_NUMBER)]

    return sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes),
                           tag_metas=sly.TagMetaCollection(tags))


def load_model(weights_path, imgsz=640, device='cpu'):
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights_path, map_location=device)  # load FP32 model

    try:
        configs_path = os.path.join(Path(weights_path).parents[0], 'opt.yaml')
        with open(configs_path, 'r') as stream:
            cfgs_loaded = yaml.safe_load(stream)
    except:
        cfgs_loaded = None

    if hasattr(model, 'module') and hasattr(model.module, 'img_size'):
        imgsz = model.module.img_size[0]
    elif hasattr(model, 'img_size'):
        imgsz = model.img_size[0]
    elif cfgs_loaded is not None and cfgs_loaded['img_size']:
        imgsz = cfgs_loaded['img_size'][0]
    else:
        sly.logger.warning(f"Image size is not found in model checkpoint. Use default: {IMG_SIZE}")
        imgsz = IMG_SIZE
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    return model, half, device, imgsz, stride


def inference(model, half, device, imgsz, stride, image: np.ndarray, meta: sly.ProjectMeta, conf_thres=0.25, iou_thres=0.45,
              augment=False, agnostic_nms=False, debug_visualization=False) -> sly.Annotation:
    names = model.module.names if hasattr(model, 'module') else model.names

    img0 = image
    # Padded resize
    img = letterbox(img0, new_shape=imgsz, stride=stride)[0]
    img = img.transpose(2, 0, 1)  # to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    inf_out = model(img, augment=augment)[0]

    # Apply NMS
    output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, agnostic=agnostic_nms)
    labels = []
    for i, det in enumerate(output):
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                top, left, bottom, right = int(xyxy[1]), int(xyxy[0]), int(xyxy[3]), int(xyxy[2])
                rect = sly.Rectangle(top, left, bottom, right)
                obj_class = meta.get_obj_class(names[int(cls)])
                tag = sly.Tag(meta.get_tag_meta(CONFIDENCE), round(float(conf), 4))
                label = sly.Label(rect, obj_class, sly.TagCollection([tag]))
                labels.append(label)

    height, width = img0.shape[:2]
    ann = sly.Annotation(img_size=(height, width), labels=labels)

    if debug_visualization is True:
        # visualize for debug purposes
        vis = np.copy(img0)
        ann.draw_contour(vis, thickness=2)
        sly.image.write("vis.jpg", vis)

    return ann.to_json()


def sliding_window_inference(model, half, device, imgsz, stride, img: np.ndarray, meta: sly.ProjectMeta, 
                             sliding_window_params, conf_thres=0.25, 
                             iou_thres=0.45, agnostic_nms=False):
    # 'img' is RGB in [H, W, C] format
    img_h, img_w = img.shape[:2]
    windowHeight = sliding_window_params.get("windowHeight", img_h)
    windowWidth = sliding_window_params.get("windowWidth", img_w)
    overlapY = sliding_window_params.get("overlapY", 0)
    overlapX = sliding_window_params.get("overlapX", 0)
    borderStrategy = sliding_window_params.get("borderStrategy", "shift_window")

    slider = SlidingWindowsFuzzy([windowHeight, windowWidth],
                                 [overlapY, overlapX],
                                 borderStrategy)

    names = model.module.names if hasattr(model, 'module') else model.names
    rectangles = []
    for window in slider.get(img.shape[:2]):
        rectangles.append(window)

    candidates = []
    slides_for_vis = []
    frame_size = None
    cropped_img_size = None
    for rect in rectangles:
        cropped_img = img[rect.top:rect.bottom+1, rect.left:rect.right+1]
        if frame_size is None:
            frame_size = cropped_img.shape[:2]
        cropped_img = letterbox(cropped_img, new_shape=imgsz, stride=stride)[0]
        if cropped_img_size is None:
            cropped_img_size = cropped_img.shape

        cropped_img = cropped_img.transpose(2, 0, 1)  # to CxHxW
        cropped_img = np.ascontiguousarray(cropped_img)
        cropped_img = torch.from_numpy(cropped_img).to(device)
        cropped_img = cropped_img.half() if half else cropped_img.float()  # uint8 to fp16/32
        cropped_img /= 255.0  
        if cropped_img.ndimension() == 3:
            cropped_img = cropped_img.unsqueeze(0)
        inf_res_base = model(cropped_img)[0][0] # inference, out coords xywh
        inf_res_base = inf_res_base[inf_res_base[..., 4] > conf_thres] # remove low box conf (not included class conf)

        if len(inf_res_base) == 0:
            slides_for_vis.append(None)
            continue
        inf_res = inf_res_base.clone()
        # prepare dets for vis
        inf_res = xywh2xyxy(inf_res)

        inf_res[:, :4] = scale_coords(cropped_img.shape[-2:], inf_res[:, :4], frame_size).round()
        
        inf_res[:, [0, 2]] += rect.left # x1, x2 to global coords
        inf_res[:, [1, 3]] += rect.top # y1, y2 to global coords
        slides_for_vis.append(inf_res)

        # prepare dets for NMS (to global image coords)
        if len(inf_res_base) > 0:
            ratio = (cropped_img.shape[-1] / frame_size[1], cropped_img.shape[-2] / frame_size[0])
            inf_res_base[:, 0] += rect.left * ratio[0] # (cropped_img.shape[1] / frame_size[0])
            inf_res_base[:, 1] += rect.top * ratio[1] # (cropped_img.shape[0] / frame_size[1])

        candidates.append(inf_res_base)
    
    if isinstance(candidates[0], np.ndarray):
        candidates = [torch.as_tensor(element) for element in candidates]
    detections = torch.cat(candidates).unsqueeze_(0)

    # get raw candidates for vis
    for i, det in enumerate(slides_for_vis):
        if det is None:
            slides_for_vis[i] = {"rectangle": rectangles[i].to_json(), "labels": []}
            continue
        labels = []
        for x1, y1, x2, y2, conf, *cls in reversed(det):
            top, left, bottom, right = y1.int().item(), x1.int().item(), y2.int().item(), x2.int().item()
            rect = sly.Rectangle(top, left, bottom, right)
            class_ind = np.argmax(cls).item()
            max_conf = np.max(cls).item()
            obj_class = meta.get_obj_class(names[class_ind])
            conf_val = round(conf.float().item(), 4) * max_conf # box_conf * class_conf
            if conf_val < conf_thres:
                continue
            tag = sly.Tag(meta.get_tag_meta(CONFIDENCE), conf_val)
            label = sly.Label(rect, obj_class, sly.TagCollection([tag]))
            labels.append(label.to_json())
        slides_for_vis[i] = {"rectangle": rectangles[i].to_json(), "labels": labels}

    # apply NMS
    detections = non_max_suppression(detections, conf_thres=conf_thres, iou_thres=iou_thres, agnostic=agnostic_nms)
    
    # get labels after NMS
    labels_after_nms = []
    for i, det in enumerate(detections):
        ratio = (img.shape[0] / frame_size[0], img.shape[1] / frame_size[1])
        size = (cropped_img_size[0] * ratio[0], cropped_img_size[1] * ratio[1])
        det[:, :4] = scale_coords(size, det[:, :4], img.shape).round()

        for *xyxy, conf, cls in reversed(det):
            top, left, bottom, right = int(xyxy[1]), int(xyxy[0]), int(xyxy[3]), int(xyxy[2])
            rect = sly.Rectangle(top, left, bottom, right)
            obj_class = meta.get_obj_class(names[int(cls)])
            tag = sly.Tag(meta.get_tag_meta(CONFIDENCE), round(float(conf), 4))
            label = sly.Label(rect, obj_class, sly.TagCollection([tag]))
            labels_after_nms.append(label)
    
    # create Annotation
    height, width = img.shape[:2]
    ann = sly.Annotation(img_size=(height, width), labels=labels_after_nms)
    for label_ind, label in enumerate(labels_after_nms):
        labels_after_nms[label_ind] = label.to_json()
    # add two last slides
    full_rect = sly.Rectangle(0, 0, img.shape[0], img.shape[1])
    all_labels_without_nms = []
    for slide in slides_for_vis:
        all_labels_without_nms.extend(slide["labels"])
    slides_for_vis.append({"rectangle": full_rect.to_json(), "labels": all_labels_without_nms})
    slides_for_vis.append({"rectangle": full_rect.to_json(), "labels": labels_after_nms})


    return ann.to_json(), slides_for_vis

