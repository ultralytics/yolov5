import torch
import numpy as np
import supervisely_lib as sly
from supervisely.io.fs import get_file_name_with_ext
import os
from pathlib import Path
import yaml
import torch.nn.functional as F
import cv2

from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
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

    meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes),
                           tag_metas=sly.TagMetaCollection(tags))
    return meta


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

    for det in output:
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

    # Apply NMS
    output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, agnostic=agnostic_nms)
    
    labels = convert_preds_to_sly(output, meta, names)

    height, width = img0.shape[:2]
    ann = sly.Annotation(img_size=(height, width), labels=labels)

    if debug_visualization is True:
        # visualize for debug purposes
        vis = np.copy(img0)
        ann.draw_contour(vis, thickness=2)
        sly.image.write("vis.jpg", vis)

    return ann.to_json()


def convert_preds_to_sly(detections, meta, names, slides_for_vis=None, rects=None):
    if slides_for_vis is not None:
        assert rects is not None
    labels = []
    for i, det in enumerate(detections):
        if slides_for_vis is None:
            if not (det is None or not len(det)):
                continue
        else:
            slide_labels = []
        for *xyxy, conf, cls in reversed(det):
            top, left, bottom, right = int(xyxy[1]), int(xyxy[0]), int(xyxy[3]), int(xyxy[2])
            rect = sly.Rectangle(top, left, bottom, right)
            obj_class = meta.get_obj_class(names[int(cls)])
            tag = sly.Tag(meta.get_tag_meta(CONFIDENCE), round(float(conf), 4))
            label = sly.Label(rect, obj_class, sly.TagCollection([tag]))
            labels.append(label)
            if slides_for_vis is not None:
                slide_labels.append(label)
        if slides_for_vis is not None:
            slides_for_vis.append({rects[i]: slide_labels})
    return labels

def sliding_window_inference(model, half, device, img: np.ndarray, meta: sly.ProjectMeta, 
                             sliding_window_params, conf_thresh=0.25, 
                             iou_thresh=0.45, agnostic_nms=False):
    # 'img' is RGB in [H, W, C] format
    img_h, img_w = img.shape[:2]
    base_img = img
    windowHeight = sliding_window_params.get("windowHeight", img_h)
    windowWidth = sliding_window_params.get("windowWidth", img_w)
    overlapY = sliding_window_params.get("overlapY", 0)
    overlapX = sliding_window_params.get("overlapX", 0)
    borderStrategy = sliding_window_params.get("borderStrategy", img_h)
    naive = sliding_window_params.get("naive", False)

    slider = SlidingWindowsFuzzy([windowHeight, windowWidth],
                                 [overlapY, overlapX],
                                 borderStrategy)

    names = model.module.names if hasattr(model, 'module') else model.names
    
    rectangles = []
    for window in slider.get(img.shape[:2]):
        rectangles.append(window)

    img = img.transpose(2, 0, 1)  # to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    candidates = []
    for rect in rectangles:
        cropped_image = img[..., rect.top:rect.bottom, rect.left:rect.right]
        inf_res = model(cropped_image)[0] # inference
        if naive:
            inf_res = inf_res if len(inf_res.shape) == 3 else np.expand_dims(inf_res, axis=0)
            inf_res = non_max_suppression(inf_res,
                                            conf_thres=conf_thresh,
                                            iou_thres=iou_thresh,
                                            agnostic=agnostic_nms)[0]
        for det in inf_res:
            det[:, :4] = scale_coords(img.shape[-2:], det[:, :4], base_img.shape).round()
        candidates.append(inf_res)
    
    if isinstance(candidates[0], np.ndarray):
        candidates = [torch.as_tensor(element) for element in candidates]
    detections = torch.cat(candidates).unsqueeze_(0)

    if not naive:
        final_detections = non_max_suppression(detections, conf_thres=conf_thresh, iou_thres=iou_thresh, agnostic=agnostic_nms)
    
        # Convertion to Supervisely Annotation
        slides_for_vis = []
        labels = convert_preds_to_sly(detections, meta, names, slides_for_vis, rectangles)
        # TODO: mistake: slides without labels will be skipped
        final_labels = convert_preds_to_sly(final_detections, meta, names)
        
        # add two last slides (before NMS and after)
        full_rect = sly.Rectangle(0, 0, base_img.shape[0], base_img.shape[1])
        slides_for_vis.append({full_rect: labels})
        slides_for_vis.append({full_rect: final_labels})
    else:
        slides_for_vis = []
        labels = convert_preds_to_sly(detections, meta, names, slides_for_vis, rectangles)
        # add two last slides (both are the same)
        full_rect = sly.Rectangle(0, 0, base_img.shape[0], base_img.shape[1])
        slides_for_vis.append({full_rect: labels})
        slides_for_vis.append({full_rect: labels})

    # create Annotation
    height, width = base_img.shape[:2]
    ann = sly.Annotation(img_size=(height, width), labels=final_labels)

    return ann.to_json(), slides_for_vis