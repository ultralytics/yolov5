"""
This module implements TorchCAM module for YOLOv5 in order to see where the model is attenting to.
Requirements: pip install torchcam

Testing:    !python explainer.py --source data/images/zidane.jpg --verbose
or:         from explainer import run; run(source='data/images/zidane.jpg')
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
# from torchcam.methods import cam

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import print_args,check_file
from utils.torch_utils import select_device

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages,IMG_FORMATS, VID_FORMATS
from utils.general import check_img_size


def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        method='EigenCAM', # the method for interpreting the results
        layer=-2 ,
        class_names= None, # list of class names to use for CAM methods
        objectness_thres=0.1, # threshold for objectness
        imgsz=(640, 640),  # inference size (height, width)
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        nosave=False,  # do not save images/videos
        dnn=False,  # use OpenCV DNN for ONNX inference
        half=False,  # use FP16 half-precision inference
        verbose=False,  # verbose output
        vid_stride=1,  # video frame-rate stride
):
    # copied from detect.py
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    if is_url and is_file:
        source = check_file(source)  # download
    # copied from detect.py

    use_cuda = len(device) > 0 # for now we can not choose GPU device
    device = select_device(device)
    
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, pt = model.stride, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    model.requires_grad_(True)
    # model.eval() # not sure about this! 
    dataset =LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # reverse key,values pairs since we to index with reverse 
    model_classes =dict((v,k) for k,v in model.names.items())
    class_idx = [model_classes[item] for item in class_names]

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        
        prediction, heads = model(im) 
        return prediction, heads
        cam_image = explain(method=method,model= model, image=im, layer=layer, 
                    classes=class_idx, objectness_thres=objectness_thres,use_cuda=use_cuda)

        # for now, we only support one image at a time
        # then we should save the image in a file
        return cam_image
    


def parseopt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--method',
                        type=str,
                        default='EigenCAM',
                        help="the method to use for interpreting the feature maps")
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
