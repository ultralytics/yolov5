"""
Script for serving.
"""
import os
import random

import numpy as np
import torch
from PIL import Image
from flask import Flask, request

from utils_image import encode_image, decode_image
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, plot_one_box

MODEL_DIR = "/artefact/"
if os.path.exists("weights/best.pt"):
    MODEL_DIR = "weights/"

DEVICE = torch.device("cpu")
MODEL = attempt_load(MODEL_DIR + "best.pt", map_location=DEVICE)
IMGSZ = check_img_size(416, s=MODEL.stride.max())

# Get names and colors
NAMES = MODEL.module.names if hasattr(MODEL, 'module') else MODEL.names
COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(NAMES))]


def process_img(img0):
    # Padded resize
    img = letterbox(img0, new_shape=IMGSZ)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(DEVICE)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0)
    return img


def predict(encoded_img, conf_thres, iou_thres, agnostic_nms, augment, classes=None):
    """Predict function for image input."""
    img0 = decode_image(encoded_img)
    img = process_img(img0)

    # Inference
    pred = MODEL(img, augment=augment)[0]

    # Apply NMS
    det = non_max_suppression(pred, conf_thres, iou_thres,
                              classes=classes, agnostic=agnostic_nms)[0]

    output_dict = process_detections(det, img.shape[2:], img0)
    return output_dict, encode_image(Image.fromarray(img0))


def process_detections(det, img_shape, img0):
    """Process detections."""
    output_dict = {"shellfishDetection": list()}
    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    if det is not None and len(det):
        # Rescale boxes from img_size to img0 size
        det[:, :4] = scale_coords(img_shape, det[:, :4], img0.shape).round()

        # Write results
        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            output_dict["shellfishDetection"].append({
                "boundingPoly": {
                    "normalizedVertices": [{
                         "x": xywh[0],
                         "y": xywh[1],
                         "width": xywh[2],
                         "height": xywh[3],
                    }]
                },
                "name": NAMES[int(cls)],
                "score": float(conf.numpy()),
            })

            label = '%s %.2f' % (NAMES[int(cls)], conf)
            plot_one_box(xyxy, img0, label=label, color=COLORS[int(cls)], line_thickness=3)
    return output_dict


# pylint: disable=invalid-name
app = Flask(__name__)


@app.route("/", methods=["POST"])
def get_prob():
    """Returns probability."""
    req_json = request.json
    encoded_img = req_json["encoded_img"]
    conf_thres = req_json.get("conf_thres") or 0.6
    iou_thres = req_json.get("iou_thres") or 0.5
    agnostic_nms = req_json.get("agnostic_nms") or True
    augment = req_json.get("augment") or True
    output_dict, encoded_img = predict(encoded_img, conf_thres, iou_thres, agnostic_nms, augment)
    return {"output_dict": output_dict, "encoded_img": encoded_img}


def main():
    """Starts the Http server"""
    app.run()


if __name__ == "__main__":
    main()
