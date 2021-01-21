import os
import cv2
import torch
from PIL import Image
import numpy as np
import json
import supervisely_lib as sly


import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized





my_app = sly.AppService()

TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
image_id = 725268

meta: sly.ProjectMeta = None

remote_weights_path = "/yolov5_train/coco128_002/2278_072/weights/best.pt"
local_weights_path = None
model = None


def download_weights():
    global local_weights_path
    local_weights_path = remote_weights_path
    sly.fs.ensure_base_path(local_weights_path)
    my_app.public_api.file.download(TEAM_ID, remote_weights_path, local_weights_path)


def init_output_meta(model):
    global meta
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

    obj_classes = []
    for name, color in zip(names, colors):
        obj_classes.append(sly.ObjClass(name, sly.Rectangle, color))

    meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))


@my_app.callback("get_output_classes_and_tags")
@sly.timeit
def get_output_classes_and_tags(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    my_app.send_response(request_id, data=meta.to_json())


@my_app.callback("get_session_info")
@sly.timeit
def get_session_info(api: sly.Api, task_id, context, state, app_logger):
    info = {
        "app": "YOLO v5 serve",
        "model": remote_weights_path,
        "custom-field": "hello!"
    }
    request_id = context["request_id"]
    my_app.send_response(request_id, data=info)


@my_app.callback("inference_image_id")
@sly.timeit
def inference_image_id(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_id = state["image_id"]
    debug_visualization = state.get("debug_visualization", False)
    image = api.image.download_np(image_id)  # RGB image
    ann_json = inference(image, debug_visualization)

    request_id = context["request_id"]
    my_app.send_response(request_id, data=ann_json)


def detect(weights, image, device='cpu'):
    # Initialize
    imgsz = 640  #@TODO: get from model
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


def inference(image: np.ndarray, debug_visualization=False) -> sly.Annotation:
    height, width = image.shape[:2]
    ann = sly.Annotation(img_size=(width, height))

    pred = model(image)

    pred.print()  # print results to screen
    pred.show()  # display results
    pred.save()  # save as results1.jpg, results2.jpg... etc.

    for i, det in enumerate(pred):  # detections per image
        print(i)
        print(det)
        # if det is not None and len(det):
        #     # Rescale boxes from img_size to im0 size
        #     #det[:, :4] = scale_coords(image.shape[2:], det[:, :4], image_np_0.shape).round()
        #     # Write results
        #     for *xyxy, conf, cls in reversed(det):
        #         top, left, bottom, right = int(xyxy[1]), int(xyxy[0]), int(xyxy[3]), int(xyxy[2])
        #         new_geom = sly.Rectangle(top, left, bottom, right)
        #         new_obj_class = new_collection.get(names[int(cls)])
        #         if confidence_tag_name is not None:
        #             tag = sly.Tag(tag_meta, round(float(conf), 2))  # tensor.item()
        #             new_tag = sly.TagCollection([tag])
        #             new_label = sly.Label(new_geom, new_obj_class, new_tag)
        #         else:
        #             new_label = sly.Label(new_geom, new_obj_class)
        #         ann = ann.add_label(new_label)




    # if debug_visualization is True:
    #     # visualize for debug purposes
    #     vis_filled = np.zeros((height, width, 3), np.uint8)
    #     ann.draw(vis_filled)
    #     vis = cv2.addWeighted(image, 1, vis_filled, 0.5, 0)
    #     ann.draw_contour(vis, thickness=5)
    #     sly.image.write("vis.jpg", vis)

    return ann.to_json()


def debug_inference():
    image = sly.image.read("./data/images/bus.jpg")  # RGB
    ann = inference(image)
    print(json.dumps(ann, indent=4))


# def deploy_model():
#     results = model(img1)
#
#     # Results
#     results.print()  # print results to screen
#     results.show()  # display results
#     results.save()  # save as results1.jpg, results2.jpg... etc.
#
#     x = 10
#     x += 1



def main():
    download_weights()

    global model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=local_weights_path)

    init_output_meta(model)
    debug_inference()

    x = 10
    x += 1

#@TODO: save image size to model
#@TODO: fix serve template - debug_inference
#@TODO: or another app serve_cpu?
#@TODO: deploy on custom device: cpu/gpu
if __name__ == "__main__":
    sly.main_wrapper("main", main)