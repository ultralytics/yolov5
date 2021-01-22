import torch
import numpy as np
import supervisely_lib as sly

from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.datasets import letterbox


CONFIDENCE = "confidence"


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
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    return model, half, device, imgsz


def inference(model, half, device, image: np.ndarray, imgsz, meta: sly.ProjectMeta, conf_thres=0.25, iou_thres=0.45,
              augment=False, agnostic_nms=False, debug_visualization=False) -> sly.Annotation:
    names = model.module.names if hasattr(model, 'module') else model.names

    img0 = image
    # Padded resize
    img = letterbox(img0, new_shape=imgsz)[0]
    img = img.transpose(2, 0, 1)  # to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=augment)[0]

    # Apply NMS
    labels = []
    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=agnostic_nms)
    for i, det in enumerate(pred):
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
    ann = sly.Annotation(img_size=(width, height), labels=labels)

    if debug_visualization is True:
        # visualize for debug purposes
        vis = np.zeros((height, width, 3), np.uint8)
        ann.draw_contour(vis, thickness=2)
        sly.image.write("vis.jpg", vis)

    return ann.to_json()




