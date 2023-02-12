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

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from pytorch_grad_cam import AblationCAM, EigenCAM, FullGrad, GradCAM, GradCAMPlusPlus, HiResCAM, ScoreCAM, XGradCAM
from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image

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

def yolo_reshape_transform(x):
    """
    The backbone outputs different tensors with different spatial sizes, from the FPN.
    Our goal here is to aggregate these image tensors, assign them weights, and then aggregate everything.
    To do that, we’re going to need to write a custom function that takes these tensors with different sizes,
    resizes them to a common shape, and concatenates them
    https://jacobgil.github.io/pytorch-gradcam-book/Class%20Activation%20Maps%20for%20Object%20Detection%20With%20Faster%20RCNN.html

    it seems that output is always the same shape in yolo. So, this is not needed.
    """
    return x


class YOLOBoxScoreTarget():
    """ For every original detected bounding box specified in "bounding boxes",
        assign a score on how the current bounding boxes match it,
            1. In IOU
            2. In the classification score.
        If there is not a large enough overlap, or the category changed,
        assign a score of 0.

        The total score is the sum of all the box scores.
    """

    def __init__(self,classes,objectness_threshold):
        self.classes = set(classes)
        self.objectness_threshold = objectness_threshold

    def __call__(self, output):
        """
        here we need something which we can call backward
        https://pub.towardsai.net/yolov5-m-implementation-from-scratch-with-pytorch-c8f84a66c98b
        output structure is taken from this tutorial, it is as follows:
        "xc,yc,height, width,objectness, classes"

        so, the first item would be objectness and items after fifth element are class indexes
        """
        objectness = output[0, :, 4]
        classes = output[0, :, 5:]
        mask = torch.zeros_like(classes, dtype=torch.bool)
        for class_idx in self.classes:
            mask[:, class_idx] = True

        # we have to see if below line has any effect  
        mask[objectness<self.objectness_threshold] = False
        score = classes[mask] # + objectness[mask]
        return score.sum()
        

def extract_eigenCAM(model, image, layer= -2):
    """
    eigenCAM doesn't acutally needs YOLOBoxScoreTarget. It doesn't call it.
    """
    target_layers = [model.model.model[layer]]
    cam = EigenCAM(model, target_layers, use_cuda=False)
    # transform = transforms.ToTensor()
    # tensor = transform(raw_image_fp).unsqueeze(0)
    breakpoint()
    grayscale_cam = cam(image)[0, :, :]

    cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    return cam_image


def extract_gradCAM(model, image,layer,classes, objectness_thres):
    
    #target_layers = [model.model[-1].m[0]]
    target_layers =[model.model.model[layer]]
    # target_layers= [model.model.model.model[layer]]
    targets = [YOLOBoxScoreTarget(classes=classes, objectness_threshold=objectness_thres)]
    cam = GradCAM(model, target_layers, use_cuda=torch.cuda.is_available(), reshape_transform=yolo_reshape_transform)

    # transform = transforms.ToTensor()
    # tensor = transform(image).unsqueeze(0)

    # grayscale_cam = cam(tensor, targets=targets)
    grayscale_cam= cam(image,targets=targets)
    # Take the first image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    # And lets draw the boxes again:
    #image_with_bounding_boxes = draw_boxes(prediction, cam_image)
    #cam_image = Image.fromarray(image_with_bounding_boxes)
    return cam_image

def explain(method:str, model,image,layer,classes, objectness_thres:float):
    cam_image = None
    if method.lower()=='gradcam':
        cam_image=extract_gradCAM(model,image,layer,classes,objectness_thres)
    elif method.lower()=='eigencam':
        cam_image= extract_eigenCAM(model,image,layer)
    else:
        raise NotImplementedError('The method that you requested has not yet been implemented')

    return cam_image

def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        method='EigenCAM', # the method for interpreting the results
        layer=-2 ,
        classes= None, # list of class_idx to use for CAM methods
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

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    model.requires_grad_(True)
    # model.eval() # not sure about this! 
    dataset =LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = model(im)
        print(pred[0].shape, pred[1].shape)
    # image_file = Image.open(source, 'r')
    # raw_image = Image.Image.resize(image_file, (640, 384))

    # tensor_image = transforms.ToTensor()(raw_image).unsqueeze(dim=0)
    # results = model(tensor_image)
    # print(logits[0].shape)
    #results.save()

    # if verbose:
    #     print('\n', logits.shape.pandas().xyxy, '\n')
    #     print('\n', results.xyxy, '\n')

    # if verbose:
    #     print('\n', 'model layers: you have to choose a layer or some layers to explain them')
    #     layer_number = 1
    #     for k, v in model.model.model.model.named_parameters():
    #         #print(k)
    #         pass

    # raw_image_fp = np.array(raw_image, np.float32)
    # raw_image_fp = raw_image_fp / 255
    # cam_image = explain(method=method,model= model, image=raw_image, layer=layer, 
    #             classes=classes, objectness_thres=objectness_thres)

    # # Image.Image.show(Image.fromarray(cam_image))
    # return Image.fromarray(cam_image)


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
