# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2

import wandb

import numpy as np
import torch

from utils.general import xyxy2xywhn, scale_coords, xyxy2xywh, clip_coords
from mojo_val import plot_predictions_and_labels

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.general import (
    check_dataset,
    check_file,
    check_img_size,
    non_max_suppression,
    set_logging,
    increment_path,
    colorstr,
)
from utils.torch_utils import select_device
import val

from aisa_utils.dl_utils.utils import (
    plot_object_count_difference_ridgeline,
    make_video_results,
    plot_object_count_difference_line,
)


def get_images_and_labels(data):
    image_paths = list(Path(data).glob("*.jpg")) + list(Path(data).glob("*.png"))
    images = []
    labels = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), 0)
        images.append(image)
        label_path = (
            image_path.parents[2]
            / "labels"
            / image_path.parent.stem
            / f"{image_path.stem}.txt"
        )
        if label_path.is_file():
            label = []
            with label_path.open("r") as f:
                for line in f.readlines():
                    line = line.split()
                    line = [int(line[0])] + list(map(float, line[1:]))
                    label.append(line)
            labels.append(label)
        else:
            raise Exception(f"Missing label {label_path}")
    return images, labels


@torch.no_grad()
def mojo_test(
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    save_txt=False,  # save results to *.txt
    project="runs/val",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    entity=None,
    test_video_root=None,
):

    device = select_device(device, batch_size=batch_size)
    print(f"data:{data}")
    data_dict = check_dataset(data)
    print(f"data_dict:{data_dict}")
    # Trainloader
    images, labels = get_images_and_labels(data_dict["test"])

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir
    run_id = torch.load(weights[0]).get("wandb_id")

    wandb_run = wandb.init(
        id=run_id, project=project, entity=entity, resume="allow", allow_val_change=True
    )

    results, maps, t, extra_stats = val.run(
        data,
        weights=weights,  # model.pt path(s)
        batch_size=batch_size,  # batch size
        imgsz=imgsz,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task="test",  # train, val, test, speed or study
    )
    total_inference_time = np.sum(t)
    print(f"total_inference_time={total_inference_time:.1f}ms")
    wandb_run.log({f"mojo_test/test_metrics/mp": results[0]})
    wandb_run.log({f"mojo_test/test_metrics/mr": results[1]})
    wandb_run.log({f"mojo_test/test_metrics/map50": results[2]})
    wandb_run.log({f"mojo_test/test_metrics/map": results[3]})
    wandb_run.log({f"mojo_test/test_metrics/inference_time": total_inference_time})

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check image size

    # Half
    half = device.type != "cpu"  # half precision only supported on CUDA
    if half:
        model.half()

    def video_prediction_function(
        frame_array, iou_thres_nms=0.45, conf_thres_nms=0.001
    ):
        n_frames = len(frame_array)
        preds = []
        for i in range(0, n_frames, batch_size):
            frames = []
            for frame in frame_array[i : min(i + batch_size, n_frames)]:
                from utils.datasets import letterbox

                img = letterbox(frame, new_shape=(imgsz, imgsz))[0]
                img = np.array([img, img, img])
                img = np.ascontiguousarray(img)
                frames.append(img)
            frames = np.array(frames)

            # Convert img to torch
            img = torch.from_numpy(frames).to(device)
            img = (
                img.half() if device.type != "cpu" else img.float()
            )  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            pred = model(img, augment=False)[0]
            # Apply NMS
            pred = non_max_suppression(
                pred, iou_thres=iou_thres_nms, conf_thres=conf_thres_nms
            )
            _ = []

            for j, det in enumerate(pred):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], frame_array[i + j].shape
                )
                det = det.cpu().numpy()
                det[:, :4] = xyxy2xywhn(
                    det[:, :4],
                    w=frame_array[i + j].shape[1],
                    h=frame_array[i + j].shape[0],
                )
                _.append(det)
            preds += _
        return preds

    extra_plots = dict()
    preds_iou_thres = dict()
    for iout in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]:
        preds_iou_thres[iout] = video_prediction_function(images, iou_thres_nms=iout)
    fig_line = plot_object_count_difference_line(labels, preds_iou_thres)

    preds = preds_iou_thres[0.45]
    fig, suggested_threshold = plot_object_count_difference_ridgeline(labels, preds)

    extra_plots["object_count_difference"] = fig
    extra_plots["object_count_difference_continuous"] = fig_line

    extra_plots.update(plot_predictions_and_labels(extra_stats))

    print(f"suggested_threshold={suggested_threshold}")
    for plot_key in extra_plots:
        wandb_run.log({f"mojo_test/extra_plots/{plot_key}": extra_plots[plot_key]})

    if test_video_root is not None:
        for video_path in Path(test_video_root).rglob("*.avi"):
            output_video_path, jitter_plot = make_video_results(
                video_path, lambda x: video_prediction_function(x, suggested_threshold)
            )

            wandb_run.log(
                {
                    f"mojo_test/extra_videos/{output_video_path.name}": wandb.Video(
                        str(output_video_path), fps=60, format="mp4"
                    )
                }
            )
            wandb_run.log(
                {f"mojo_test/extra_plots/{output_video_path.name}_jitter": jitter_plot}
            )

    return None


def parse_opt():
    parser = argparse.ArgumentParser(prog="mojo_test.py")
    parser.add_argument(
        "--data", type=str, default="data/coco128.yaml", help="dataset.yaml path"
    )
    parser.add_argument(
        "--weights", nargs="+", type=str, default="yolov5s.pt", help="model.pt path(s)"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        type=int,
        default=640,
        help="inference size (pixels)",
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--project", default="runs_test", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument("--entity", default=None, help="W&B entity")
    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    return opt


def main(opt):
    set_logging()
    print(colorstr("mojo test: ") + ", ".join(f"{k}={v}" for k, v in vars(opt).items()))
    mojo_test(**vars(opt))


if __name__ == "__main__":
    # weights = Path(r"D:\Nanovare\dev\yolov5\wandb_mod_tests\exp\weights\best.pt")
    opt = parse_opt()
    main(opt)
