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

from models.experimental import attempt_load
from utils.general import print_args


def yolo_reshape_transform(x):
    """
    The backbone outputs different tensors with different spatial sizes, from the FPN.
    Our goal here is to aggregate these image tensors, assign them weights, and then aggregate everything.
    To do that, we’re going to need to write a custom function that takes these tensors with different sizes,
    resizes them to a common shape, and concatenates them
    https://jacobgil.github.io/pytorch-gradcam-book/Class%20Activation%20Maps%20for%20Object%20Detection%20With%20Faster%20RCNN.html
    """
    return x
    target_size = x['pool'].size()[-2:]
    activations = []
    for key, value in x.items():
        activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    return activations


class YOLOBoxScoreTarget():
    """ For every original detected bounding box specified in "bounding boxes",
        assign a score on how the current bounding boxes match it,
            1. In IOU
            2. In the classification score.
        If there is not a large enough overlap, or the category changed,
        assign a score of 0.

        The total score is the sum of all the box scores.
    """

    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels  # true_labels
        self.bounding_boxes = bounding_boxes  # true_bbox
        self.iou_threshold = iou_threshold

    def __call__(self, output):
        score = 0
        for class_idx in self.labels:
            score += output[...,class_idx]
        return score

        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()

        if len(predictions) == 0:
            return output

        breakpoint()
        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()

            bbox_pred = predictions[['xmin', 'ymin', 'xmax', 'ymax']].to_numpy()
            ious = torchvision.ops.box_iou(box, torch.tensor(bbox_pred))
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and predictions["class"][index] == label:
                score = ious[0, index] + predictions["confidence"][index]
                output = output + score
        return output


def extract_eigenCAM(model, raw_image_fp, layer= -2):
    """
    eigenCAM doesn't acutally needs YOLOBoxScoreTarget. It doesn't call it.
    to see eigenCAM layer changes, you have to restart COLAB completely
    """
    target_layers = [model.model.model.model[layer]]
    cam = EigenCAM(model, target_layers, use_cuda=False)
    transform = transforms.ToTensor()
    tensor = transform(raw_image_fp).unsqueeze(0)

    grayscale_cam = cam(tensor)[0, :, :]

    cam_image = show_cam_on_image(raw_image_fp, grayscale_cam, use_rgb=True)
    return cam_image


def extract_gradCAM(model, raw_image_fp,layer):
    true_boxes = np.array([
        [360,20, 570,380 ],
        [60,100,400,380],
        [200,234,249,382],
    ])

    true_labels = [0,0,27]

    target_layers = [model.model[-1].m[0]]
    targets = [YOLOBoxScoreTarget(labels=true_labels, bounding_boxes=true_boxes)]
    cam = GradCAM(model, target_layers, use_cuda=torch.cuda.is_available(), reshape_transform=yolo_reshape_transform)

    transform = transforms.ToTensor()
    tensor = transform(raw_image_fp).unsqueeze(0)

    grayscale_cam = cam(tensor, targets=targets)
    # Take the first image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(raw_image_fp, grayscale_cam, use_rgb=True)
    # And lets draw the boxes again:
    #image_with_bounding_boxes = draw_boxes(prediction, cam_image)
    #cam_image = Image.fromarray(image_with_bounding_boxes)
    return cam_image


def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        iou_thres=0.45,  # NMS IOU threshold
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        method='EigenCAM',  # the method for interpreting the results
        layer=-2 ,
        verbose=False,  # verbose output
):
    # model = torch.hub.load(
    #     'ultralytics/yolov5',
    #     'yolov5s',
    #     #autoshape=False #because otherwise I have to resize the image, I just don't know for now
    # )
    model = attempt_load('yolov5s.pt')
    model.requires_grad_(True)
    model.eval() # not sure about this! 

    image_file = Image.open(source, 'r')
    raw_image = Image.Image.resize(image_file, (640, 384))

    tensor_image = transforms.ToTensor()(raw_image).unsqueeze(dim=0)
    preds, logits = model(tensor_image)
    print(logits[0].shape)
    #results.save()

    # if verbose:
    #     print('\n', logits.shape.pandas().xyxy, '\n')
    #     print('\n', results.xyxy, '\n')

    if verbose:
        print('\n', 'model layers: you have to choose a layer or some layers to explain them')
        layer_number = 1
        for k, v in model.model.model.model.named_parameters():
            #print(k)
            pass

    raw_image_fp = np.array(raw_image, np.float32)
    raw_image_fp = raw_image_fp / 255
    if method.lower() == 'eigencam':
        cam_image = extract_eigenCAM(model= model,raw_image_fp= raw_image_fp,layer=layer)
    elif method.lower() == 'gradcam':
        cam_image = extract_gradCAM(model=model,raw_image_fp= raw_image_fp,layer=layer)

    # Image.Image.show(Image.fromarray(cam_image))
    return Image.fromarray(cam_image)


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
