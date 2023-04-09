"""
This module implements GradCAM module for YOLOv5 in order to see where the model is attenting to.
Requirements: pip install grad-cam==1.4.6

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
from pytorch_grad_cam import (AblationCAM, EigenCAM, FullGrad, GradCAM, 
GradCAMPlusPlus, HiResCAM, ScoreCAM, XGradCAM, EigenGradCAM, GradCAMElementWise, LayerCAM,RandomCAM)
from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import print_args,check_file
from utils.torch_utils import select_device

from models.common import DetectMultiBackend, AutoShape
from utils.dataloaders import LoadImages,IMG_FORMATS, VID_FORMATS
from utils.general import check_img_size


def yolo_reshape_transform(x):
    """
    The backbone outputs different tensors with different spatial sizes, from the FPN.
    Our goal here is to aggregate these image tensors, assign them weights, and then aggregate everything.
    To do that, we are going to need to write a custom function that takes these tensors with different sizes,
    resizes them to a common shape, and concatenates them
    https://jacobgil.github.io/pytorch-gradcam-book/Class%20Activation%20Maps%20for%20Object%20Detection%20With%20Faster%20RCNN.html

    it seems that output is always the same shape in yolo. So, this is not needed.
    """
    return x


class YOLOBoxScoreTarget():
    """ This way we see all boxes. then we select boxes that have a good objectness score
    then we filter out classes and select the classes that we want to attend to. 

    At the end, we sum out of all these. 

    This is not a standard approach. This is somewhat similar to what 
    https://github.com/pooya-mohammadi/yolov5-gradcam
    has done. 

    Here the problem is that we are taking a lot of attention to overlapping boxes.
    This should not be the case. 
    """

    def __init__(self,classes,objectness_threshold):
        self.classes = set(classes)
        self.objectness_threshold = objectness_threshold

    def __call__(self, output):
        """
        here we need something which we can call backward
        https://pub.towardsai.net/yolov5-m-implementation-from-scratch-with-pytorch-c8f84a66c98b
        output structure is taken from this tutorial, it is as follows:

        first item is important, second item contains three arrays which contain prediction from three heads
        we would use the first array as it is the final prediction.
        pred = output[0] 
        Here, we take the first item as the second item contains predictions from three heads. Also, each head dimension would be different 
        as we have different dimensions per head. 

        "xc,yc,height, width,objectness, classes"
        so, the forth item would be objectness and items after fifth element are class indexes
        """
        if len(output.shape)==2:
            output = torch.unsqueeze(output,dim=0)

        assert len(output.shape) == 3
         # first item would be image index, number of images
         # second: number of predictions 
         # third:  predicited bboxes 
        objectness = output[:,:, 4] 
        classes = output[:,:, 5:] 
        mask = torch.zeros_like(classes, dtype=torch.bool)
        for class_idx in self.classes:
            mask[:,:, class_idx] = True
 
        mask[objectness<self.objectness_threshold] = False
        score = classes[mask] # + objectness[mask]
        return score.sum()

class YOLOBoxScoreTarget2():
    """ For every original detected bounding box specified in "bounding boxes",
        assign a score on how the current bounding boxes match it,
            1. In IOU
            2. In the classification score.
        If there is not a large enough overlap, or the category changed,
        assign a score of 0.

        The total score is the sum of all the box scores.
    """

    def __init__(self,objectness_threshold,predicted_bbox):
        self.objectness_threshold = objectness_threshold
        self.predicted_bbox = predicted_bbox

    def __call__(self, output):
        """
        here we need something which we can call backward
        https://pub.towardsai.net/yolov5-m-implementation-from-scratch-with-pytorch-c8f84a66c98b
        output structure is taken from this tutorial, it is as follows:

        first item is important, second item contains three arrays which contain prediction from three heads
        we would use the first array as it is the final prediction.
        pred = output[0] 
        Here, we take the first item as the second item contains predictions from three heads. Also, each head dimension would be different 
        as we have different dimensions per head. 

        "xc,yc,height, width,objectness, classes"
        so, the forth item would be objectness and items after fifth element are class indexes
        """
        if len(output.shape)==2:
            output = torch.unsqueeze(output,dim=0)

        assert len(output.shape) == 3
         # first dimension would be image index, number of images
         # second: number of predictions 
         # third:  predicited bboxes 
        
        bboxes = output[:,:,:4] # this is formatted differently as we need. 
        bboxes_processed = torch.zeros_like(bboxes)
        x,y,w,h = output[:,:,0],output[:,:,1],output[:,:,2],output[:,:,3]
        bboxes_processed[:,:,0] = (x-w/2)*640 # image witdth and height
        bboxes_processed[:,:,1] = (y-h/2)*640
        bboxes_processed[:,:,2] = (x+w/2)*640
        bboxes_processed[:,:,3] = (y+h/2)*640
        
        score = torch.tensor([0.0],requires_grad=True)
        iou_scores = torchvision.ops.box_iou(bbox[:,:4],bboxes_processed[0])
        breakpoint()
        for bbox in self.predicted_bbox:
            # bbox format: x1, y1, x2, y2, confidence, class_idx
            ious=torchvision.ops.box_iou(bbox[None,:4],bboxes_processed[0])
            value,index = ious.max(axis=1)
            confidence, class_idx=bbox[4], int(bbox[5])
            iou_threshold = 0.001

            #predicted_class = output[0,index, 5:].argmax(axis=1)
            if value > iou_threshold: #and predicted_class > 0.1:
                print(output[0,index,4],output[0,index,5+class_idx])
                score = score + output[0,index,5+class_idx] #+ output[0,index, 5+predicted_class]
        
        print(score)
        return score



def extract_CAM(method, model: torch.nn.Module,model_output,image,layer:int,classes, objectness_score:float, use_cuda:bool,
    **kwargs):
    target_layers =[model.model.model.model[layer]]
    #targets = [YOLOBoxScoreTarget(classes=classes, objectness_threshold=objectness_score)]

    bbox = model_output.pandas().xyxy[0]
    bbox_torch = torch.tensor(bbox.drop('name',axis=1).values)

    targets = [YOLOBoxScoreTarget2(objectness_threshold=objectness_score,
                                    predicted_bbox=bbox_torch)]
    
    cam = method(model, target_layers, use_cuda=use_cuda, 
            reshape_transform=yolo_reshape_transform, **kwargs)
    grayscale_cam= cam(image,targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    print(grayscale_cam.max())
    fixed_image = np.array(image[0]).transpose(1,2,0)
    cam_image = show_cam_on_image(fixed_image, grayscale_cam, use_rgb=True)
    # And lets draw the boxes again:
    #image_with_bounding_boxes = draw_boxes(prediction, cam_image)
    return cam_image

def explain(method:str, model,model_output,image,layer:int,classes, objectness_thres:float,use_cuda:bool):
    cam_image = None
    method_obj = None
    extra_arguments = {}

    if method.lower()=='GradCAM'.lower():
        method_obj = GradCAM
    elif method.lower()=='EigenCAM'.lower():
        method_obj = EigenCAM
    elif method.lower()=='EigenGradCAM'.lower():
        method_obj = EigenGradCAM
    elif method.lower()=='GradCAMPlusPlus'.lower():
        method_obj = GradCAMPlusPlus
    elif method.lower()=='XGradCAM'.lower():
        method_obj = XGradCAM
    elif method.lower()=='HiResCAM'.lower():
        method_obj= HiResCAM
    # elif method.lower()=='FullGrad'.lower():
    #     method_obj= FullGrad
    elif method.lower()=='ScoreCAM'.lower():
        method_obj= ScoreCAM
    # elif method.lower()=='AblationCAM'.lower():
    #     extra_arguments = {
    #         'ablation_layer': None,
    #         'batch_size': 32, 
    #         'ratio_channels_to_ablate': 1.0 }
    #     method_obj= AblationCAM
    elif method.lower()=='GradCAMElementWise'.lower():
        method_obj= GradCAMElementWise
    elif method.lower()=='LayerCAM'.lower():
        method_obj= LayerCAM
    elif method.lower()=='RandomCAM'.lower():
        # this is not an actual method. It is random
        method_obj= RandomCAM
    else:
        raise NotImplementedError('The method that you requested has not yet been implemented')

    cam_image=extract_CAM(method_obj,model,model_output,image,layer,classes,
            objectness_thres,use_cuda, **extra_arguments)
    return cam_image


class YoloOutputWrapper(torch.nn.Module):
    """
    Main purpose of using this method is to eliminate the second argument in YOLO output. 
    """
    def __init__(self, model):
        super(YoloOutputWrapper, self).__init__()
        self.model = model
    
    def forward(self, x):
        """
        first one is a 3 dim array which contains predictions
        second one is a list of heads with their corresponding predictions
        """
        total_prediction, _ = self.model(x)
        return total_prediction

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
    autoshaped_model = AutoShape(DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half))

    stride, pt = model.stride, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    model.requires_grad_(True)
    # model.eval() # not sure about this! 
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # reverse key,values pairs since we to index with reverse 
    model_classes =dict((v,k) for k,v in model.names.items())
    class_idx = [model_classes[item] for item in class_names]

    for _, im, _,_,_ in dataset:
        output = autoshaped_model(im)

        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        
        
        model = YoloOutputWrapper(model)
        _ = model(im) 
        # here we use the output from autoshaped model since we need to know bbox information

        cam_image = explain(method=method,model= model,model_output=output, image=im, layer=layer, 
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
