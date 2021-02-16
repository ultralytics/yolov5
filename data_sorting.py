import argparse
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size,
    check_requirements,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    strip_optimizer,
    set_logging,
    increment_path,
)
from utils.torch_utils import select_device, load_classifier


def sort_data(save_img=True):
    source, weights, view_img, save_txt, imgsz = (
        opt.source,
        opt.weights,
        opt.view_img,
        opt.save_txt,
        opt.img_size,
    )
    webcam = (
        source.isnumeric()
        or source.endswith(".txt")
        or source.lower().startswith(("rtsp://", "rtmp://", "http://"))
    )

    # Directories
    save_dir = Path(
        increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    )  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name="resnet101", n=2)  # initialize
        modelc.load_state_dict(
            torch.load("weights/resnet101.pt", map_location=device)["model"]
        ).to(device).eval()

    # Set Dataloader
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names

    # Run inference
    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )  # run once

    count_for_save = 0
    for path, img, im0s, _ in tqdm(dataset):
        persons = False
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes=opt.classes,
            agnostic=opt.agnostic_nms,
        )

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        count_for_save += 1
        # Process detections
        for _, det in enumerate(pred):  # detections per image

            _, _, im0, _ = path, "", im0s, getattr(dataset, "frame", 0)
            save_path = (
                str(save_dir) + "/" + opt.name + "_" + str(count_for_save) + ".jpg"
            )  # img.jpg

            if len(det) != 0:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for _, conf, cls in reversed(det):

                    if save_img or view_img:
                        label = f"{names[int(cls)]} {conf:.2f}"
                        lab = label.split(" ")

                        if lab[0] == "person" or lab[0] == "persons":
                            persons = True

        if count_for_save % 20 == 0 and persons is False:
            cv2.imwrite(save_path, im0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", nargs="+", type=str, default="yolov5s.pt", help="model.pt path(s)"
    )
    parser.add_argument(
        "--source", type=str, default="data/images", help="source"
    )  # file/folder, 0 for webcam
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--project", default="runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ["yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"]:
                sort_data()
                strip_optimizer(opt.weights)
        else:
            sort_data()
