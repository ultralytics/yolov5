import random
import numpy as np
import sys
from utils.torch_utils import *
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator
from utils.augmentations import letterbox
from models.experimental import attempt_load
import gradio as gr
import os

os.system("wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt")
os.system("wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s6.pt")


def detect(img, weights):
    gpu_id = "cpu"
    device = select_device(device=gpu_id)
    model = attempt_load(weights + '.pt', device=device)
    torch.no_grad()
    model.to(device).eval()
    half = False  # half precision only supported on CUDA
    if half:
        model.half()

    img_size = 640

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    if img is None:
        sys.exit(0)

    # Run inference
    t0 = time_sync()

    im0 = img.copy()
    img = letterbox(img, img_size, stride=int(model.stride.max()), auto=False and True)[0]
    img = np.stack(img, 0)

    img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3x416x416

    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)

    if half:
        img = img.half()
    else:
        img = img.float()  # if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    # Inference
    t1 = time_sync()
    pred = model(img, augment=False, profile=False)[0]

    # to float
    if half:
        pred = pred.float()

    # Apply NMS
    pred = non_max_suppression(pred, 0.1, 0.5, classes=None, agnostic=False)
    t2 = time_sync()
    annotator = Annotator(im0, line_width=3, example=str(names))
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        s = ''
        s += '%gx%g ' % img.shape[2:]  # print string
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f'{n:g} {names[int(c)]}s, '  # add to string

            # show results
            for *xyxy, conf, cls in det:
                label = f'{names[int(cls)]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors[int(cls)])
        im0 = annotator.result()
        # Print time (inference + NMS)
        infer_time = t2 - t1

        print('{}Done.  {}'.format(s, infer_time))

    print('Done. (%.3fs)' % (time.time() - t0))

    return im0


if __name__ == '__main__':
    gr.Interface(detect, [gr.Image(type="numpy"), gr.Dropdown(choices=["yolov5s", "yolov5s6"])],
                 gr.Image(type="numpy"), title="Yolov5", examples=[["data/images/bus.jpg", "yolov5s"]],
                 description="Gradio based demo for <a href='https://github.com/ultralytics/yolov5' style='text-decoration: underline' target='_blank'>ultralytics/yolov5</a>, new state-of-the-art for real-time object detection").launch()
