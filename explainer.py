"""
This module implements GradCAM module for YOLOv5 in order to see where the model is attenting to.
Requirements: pip install grad-cam

Testing:    !python explainer.py --source data/images/zidane.jpg --verbose
or:         from explainer import run; run(source='data/images/zidane.jpg') --verbose
"""

import argparse

import os
import sys
from pathlib import Path

from utils.general import print_args
from PIL import Image

import numpy as np
import torch
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
import torchvision.transforms as transforms

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def run(
    weights=ROOT / 'yolov5s.pt',  # model path or triton URL
    source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    iou_thres=0.45,  # NMS IOU threshold
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    verbose=False, # verbose output
):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', 
     #autoshape=False #because otherwise I have to resize the image, I just don't know for now
     )
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)  # local model
    image_file = Image.open(source,'r')
    raw_image=Image.Image.resize(image_file,(640,384))
    results = model([raw_image])
    # Results
    results.print()
    results.save()
    
    if verbose:
        print('\n',results.pandas().xyxy, '\n')
        print('\n',results.xyxy, '\n')
    
    if verbose: 
        print('\n', 'model layers: you have to choose a layer or some layers to explain them')
        layer_number = 0
        for k, v in model.model.model.model.named_parameters():
            #print(k)
            pass

    raw_image_fp = np.array(raw_image,np.float32)
    raw_image_fp = raw_image_fp / 255

    target_layers = [model.model.model.model[-2]]
    cam = EigenCAM(model, target_layers, use_cuda=False)
    transform = transforms.ToTensor()
    tensor = transform(raw_image_fp).unsqueeze(0)

    grayscale_cam = cam(tensor)[0, :, :]
    cam_image = show_cam_on_image(raw_image_fp, grayscale_cam, use_rgb=True)
    Image.show(Image.fromarray(cam_image))
    

def parseopt():
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--verbose', action='store_true', help='verbose log')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    # we should check if `grad-cam` is installed
    run(**vars(opt))

if __name__ == '__main__':
    opt = parseopt()
    main(opt)