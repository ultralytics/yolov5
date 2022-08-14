# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run classification inference on images

Usage:
    $ python classify/predict.py --weights yolov5s-cls.pt --source im.jpg
"""

import argparse
import os
import sys
from pathlib import Path

import torch.nn.functional as F
import yaml
from PIL import Image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from classify.train import imshow_cls
from utils.augmentations import classify_transforms, denormalize
from utils.general import LOGGER, check_requirements, print_args, increment_path, colorstr
from utils.torch_utils import select_device, smart_inference_mode, time_sync
from models.common import DetectMultiBackend


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s-cls.pt',  # model.pt path(s)
        source=ROOT / 'data/images/bus.jpg',  # file/dir/URL/glob, 0 for webcam
        imgsz=224,  # inference size
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        show=True,
        project=ROOT / 'runs/predict-cls',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
):
    seen, dt = 1, [0.0, 0.0, 0.0]

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # YOLOv5 classification model inference
    file = str(source)
    transforms = classify_transforms(imgsz)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=None, fp16=half)
    if len(model.names) == 1000:  # ImageNet
        with open(ROOT / 'data/ImageNet.yaml', errors='ignore') as f:
            model.names = yaml.safe_load(f)['names']  # human-readable names

    # Image
    t1 = time_sync()
    im = transforms(Image.open(file)).unsqueeze(0)
    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference
    results = model(im)
    t3 = time_sync()
    dt[1] += t3 - t2

    p = F.softmax(results, dim=1)  # probabilities
    i = p.argsort(1, descending=True)[:, :5].squeeze()  # top 5 indices
    dt[2] += time_sync() - t3
    LOGGER.info(f"image 1/1 {file}: {imgsz}x{imgsz} {', '.join(f'{model.names[j]} {p[0, j]:.2f}' for j in i)}")

    # Plot
    if show:
        imshow_cls(denormalize(im), f=save_dir / Path(file).name)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    shape = (1, 3, imgsz, imgsz)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms post-process per image at shape {shape}' % t)
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
    return p


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5n-cls.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images/bus.jpg', help='file')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='train, val image size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/predict-cls', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
