"""
# This module implements GradCAM module for YOLOv5 in order to see where the model is attenting to.
Requirements: pip install grad-cam==1.4.6

Testing:    !python explainer.py --source data/images/zidane.jpg --verbose
or:         from explainer import run; run(source='data/images/zidane.jpg') --verbose
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from pytorch_grad_cam import (AblationCAM, EigenCAM, EigenGradCAM, FullGrad, GradCAM, GradCAMElementWise,
                              GradCAMPlusPlus, HiResCAM, LayerCAM, RandomCAM, ScoreCAM, XGradCAM)
from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import AutoShape, DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
from utils.general import (check_file, check_img_size, print_args,
                        xywh2xyxy, increment_path, LOGGER)
from utils.torch_utils import select_device


def yolo_reshape_transform(x):
    """
    # The backbone outputs different tensors with different spatial sizes, from the FPN.
    Our goal here is to aggregate these image tensors, assign them weights, and then aggregate everything.
    To do that, we are going to need to write a custom function that takes these tensors with different sizes,
    resizes them to a common shape, and concatenates them
    https://jacobgil.github.io/pytorch-gradcam-book/Class%20Activation%20Maps%20for%20Object%20Detection%20With%20Faster%20RCNN.html
    it seems that output is always the same shape in yolo. So, this is not needed.
    """
    return x


class YOLOBoxScoreTarget():
    """
    This way we see all boxes.
    then we filter out classes and select the classes that we want to attend to.
    At the end, we sum out of all these.

    This is not a standard approach. This is somewhat similar to what
    https://github.com/pooya-mohammadi/yolov5-gradcam
    has done.

    Here the problem is that we are taking a lot of attention to overlapping boxes.
    This should not be the case.
    """

    def __init__(self, classes):
        self.classes = set(classes)

    def __call__(self, output):
        """
        # here we need something which we can call backward
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
        if len(output.shape) == 2:
            output = torch.unsqueeze(output, dim=0)

        assert len(output.shape) == 3
        # first item would be image index, number of images
        # second: number of predictions
        # third:  predicited bboxes

        # objectness = output[:, :, 4] this can also be used later if needed
        classes = output[:, :, 5:]
        mask = torch.zeros_like(classes, dtype=torch.bool)
        for class_idx in self.classes:
            mask[:, :, class_idx] = True

        score = classes[mask]  # + objectness[mask]
        return score.sum()


class YOLOBoxScoreTarget2():
    """ # For every original detected bounding box specified in "bounding boxes",
        assign a score on how the current bounding boxes match it,
            1. In IOU
            2. In the classification score.
        If there is not a large enough overlap, or the category changed,
        assign a score of 0.

        The total score is the sum of all the box scores.
    """

    def __init__(self, predicted_bbox, backprop, classes,device):
        """
        # Initializes the YOLOBoxScoreTarget2 module.

        Args:
            predicted_bbox: A tensor containing the predicted bounding box coordinates,
                confidence scores, and class indices.
            backprop: A string indicating which parameter to backpropagate through.
            classes: A list of class indices to consider.
        """
        self.predicted_bbox = predicted_bbox
        self.backprop = backprop
        self.classes = classes
        self.device = device

    def __call__(self, output):
        """
        here we need something which we can call backward
        https://pub.towardsai.net/yolov5-m-implementation-from-scratch-with-pytorch-c8f84a66c98b
        output structure is taken from this tutorial.

        "center_x, center_y, width, height,confidence, classes"
        so, the forth item would be confidence and items after fifth element are class indexes
        """
        if len(output.shape) == 2:
            output = torch.unsqueeze(output, dim=0)

        assert len(output.shape) == 3
        # first dimension would be image index, number of images
        # second: number of predictions
        # third:  predicited bboxes

        bboxes_processed = xywh2xyxy(output[..., :4])

        iou_scores = torchvision.ops.box_iou(self.predicted_bbox[:, :4], bboxes_processed[0])
        topk_iou_values, topk_iou_indices = iou_scores.topk(k=10, dim=-1)  # get top 10 similar boxes for each of them

        score = torch.tensor([0.0], requires_grad=True,device=self.device)

        for i, (x1, y1, x2, y2, confidence, class_idx) in enumerate(self.predicted_bbox):
            # bbox format: x1, y1, x2, y2, confidence, class_idx
            class_idx = int(class_idx)

            if class_idx not in self.classes:
                continue

            indices, values = topk_iou_indices[i], topk_iou_values[i]

            # I want to select only the relevant classes
            filtered_indices = output[0, indices, 5:].max(dim=1)[1] == class_idx
            indices = indices[filtered_indices]
            values = values[filtered_indices]

            if len(indices.size()) == 0:
                continue

            softmax_result = F.softmax(values)

            class_score = (output[0, indices, 5 + class_idx] * softmax_result).sum()
            confidence = (output[0, indices, 4] * softmax_result).sum()
            x_c = (output[0, indices, 0] * softmax_result).sum()
            y_c = (output[0, indices, 1] * softmax_result).sum()
            h = (output[0, indices, 2] * softmax_result).sum()
            w = (output[0, indices, 3] * softmax_result).sum()

            # score = score + torch.log(class_score) + torch.log(confidence)
            if self.backprop == 'class':
                score = score + torch.log(class_score)
            elif self.backprop == 'confidence':
                score = score + torch.log(confidence)
            elif self.backprop == 'class_confidence':
                score = score + torch.log(confidence * class_score)
            elif self.backprop == 'x_c':
                score = score + torch.log(x_c)
            elif self.backprop == 'y_c':
                score = score + torch.log(y_c)
            elif self.backprop == 'h':
                score = score + torch.log(h)
            elif self.backprop == 'w':
                score = score + torch.log(w)
            else:
                raise NotImplementedError('Not implemented')

        return score

def extract_CAM(method, model: torch.nn.Module, predicted_bbox, classes, backward_per_class: bool, image, layer: int,
                device, backprop_array,keep_only_topk,crop, **kwargs):
    # if we have to attend to some specific class, we will attend to it. Otherwise, attend to all present classes
    if not classes:
        classes = predicted_bbox['class'].values

    target_layers = [model.model.model[layer]]

    # targets = [YOLOBoxScoreTarget(classes=classes)]

    bbox_torch = torch.tensor(predicted_bbox.drop('name', axis=1).values
                              .astype(np.float64),device=device)

    if not backprop_array:
        backprop_array = ['class']

    cam_array = []
    use_cuda = False

    if not backward_per_class:
        for item in backprop_array:
            targets = [YOLOBoxScoreTarget2(predicted_bbox=bbox_torch, backprop=item, classes=classes,device=device)]
            cam = method(model, target_layers, use_cuda=use_cuda, reshape_transform=yolo_reshape_transform, **kwargs)
            grayscale_cam = cam(image, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            cam_array.append(grayscale_cam)
    else:
        for class_ in classes:
            for item in backprop_array:
                targets = [YOLOBoxScoreTarget2(predicted_bbox=bbox_torch, backprop=item,device=device, classes=[class_])]
                cam = method(model,
                             target_layers,
                             use_cuda=use_cuda,
                             reshape_transform=yolo_reshape_transform,
                             **kwargs)
                grayscale_cam = cam(image, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                cam_array.append(grayscale_cam)

    final_cam = sum(cam_array)

    if final_cam.max() > 0: # divide by zero
        final_cam = final_cam / final_cam.max() #normalize the result

    if 0 < keep_only_topk < 100:
        k = np.percentile(final_cam, 100-keep_only_topk)
        indices = np.where(final_cam <= k)
        final_cam[indices] = 0 

    fixed_image = np.array(image[0].cpu()).transpose(1, 2, 0)

    if crop:
        indices = np.where(final_cam > 0)
        cam_image = fixed_image.copy()
        cam_image[indices] = fixed_image.mean()
        cam_image = cam_image*255
    else: 
        cam_image = show_cam_on_image(fixed_image, final_cam, use_rgb=True)
    # And lets draw the boxes again:
    # image_with_bounding_boxes = draw_boxes(prediction, cam_image)
    # annotator = Annotator(cam_image)
    # for *box, conf, cls in bbox_torch:
    #     annotator.box_label(box,label=, color=colors(cls))

    return cam_image, final_cam


def explain(method: str, raw_model, predicted_bbox, classes, backward_per_class, image, layer: int, device,
            backprop_array,keep_only_topk,crop):
    cam_image = None
    method_obj = None
    extra_arguments = {}

    if method.lower() == 'GradCAM'.lower():
        method_obj = GradCAM
    elif method.lower() == 'EigenCAM'.lower():
        method_obj = EigenCAM
    elif method.lower() == 'EigenGradCAM'.lower():
        method_obj = EigenGradCAM
    elif method.lower() == 'GradCAMPlusPlus'.lower():
        method_obj = GradCAMPlusPlus
    elif method.lower() == 'XGradCAM'.lower():
        method_obj = XGradCAM
    elif method.lower() == 'HiResCAM'.lower():
        method_obj = HiResCAM
    # elif method.lower()=='FullGrad'.lower():
    #     method_obj= FullGrad
    elif method.lower() == 'ScoreCAM'.lower():
        method_obj = ScoreCAM
    # elif method.lower()=='AblationCAM'.lower():
    #     extra_arguments = {
    #         'ablation_layer': None,
    #         'batch_size': 32,
    #         'ratio_channels_to_ablate': 1.0 }
    #     method_obj= AblationCAM
    elif method.lower() == 'GradCAMElementWise'.lower():
        method_obj = GradCAMElementWise
    elif method.lower() == 'LayerCAM'.lower():
        method_obj = LayerCAM
    elif method.lower() == 'RandomCAM'.lower():
        # this is not an actual method. It is random
        method_obj = RandomCAM
    else:
        raise NotImplementedError('The method that you requested has not yet been implemented')
    
    try:
        cam_image,heat_map = extract_CAM(method_obj,
                                raw_model,
                                predicted_bbox,
                                classes,
                                backward_per_class,
                                image,
                                layer,
                                device=device,
                                backprop_array=backprop_array,
                                keep_only_topk=keep_only_topk,
                                crop = crop,
                                **extra_arguments)
    except Exception as e:
        # model detects nothing for image so there is no interpretabiliy heatmap!
        LOGGER.error(f'{e}')
        cam_image = image
        heat_map=torch.zeros_like(image)

    return cam_image, heat_map


class YoloOutputWrapper(DetectMultiBackend):
    """
    Main purpose of using this method is to eliminate the second argument in YOLO output.
    """
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'),
                  dnn=False, data=None, fp16=False, fuse=True):
        super().__init__(weights=weights, device=device, dnn=dnn,data=data, fp16=fp16, fuse=fuse)

    def forward(self, x):
        """
        first one is a 3 dim array which contains predictions
        second one is a list of heads with their corresponding predictions
        """
        total_prediction, _ = super().forward(x)
        return total_prediction


def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        method='GradCAM',  # the method for interpreting the results
        layer=-2,
        keep_only_topk=100, # this can be 0 to 1. it shows maximum percentage of pixels 
        # which can be used for heatmap. This is good for evaluation of heatmaps!
        class_names=[],  # list of class names to use for CAM methods
        backprop_array=[],  # list of items to do backprop! It can be class, confidence,
        backward_per_class=False,  # whether the method should backprop per each class or do it all at one backward
        crop= False, 
        imgsz=(640, 640),  # inference size (height, width)
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
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

    model = YoloOutputWrapper(weights, device=device, dnn=dnn, data=data, fp16=half)
    autoshaped_model = AutoShape(DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half))

    stride, pt = model.stride, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    model.requires_grad_(True)
    # model.eval() # not sure about this!
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # reverse key,values pairs since we to index with reverse
    model_classes = {v: k for k, v in model.names.items()}
    class_idx = [model_classes[item] for item in class_names]

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    for path, im, _, _, _ in dataset:
        processed_output = autoshaped_model(im)
        predicted_bbox = processed_output.pandas().xyxy[0]
        # list of detections, on (n,6) tensor per image [xyxy, conf, cls]

        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        _ = model(im)
        # here we use the output from autoshaped model since we need to know bbox information

        cam_image, heat_map = explain(method=method,
                            raw_model=model,
                            predicted_bbox=predicted_bbox,
                            classes=class_idx,
                            backward_per_class=backward_per_class,
                            image=im,
                            layer=layer,
                            device=device,
                            backprop_array=backprop_array,
                            keep_only_topk=keep_only_topk,
                            crop=crop)

        # for now, we only support one image at a time
        # then we should save the image in a file
        
        if save_img:
            path = Path(path)
            save_path = str(save_dir / path.name)  # im.jpg
            cv2.imwrite(save_path, cam_image)
            cv2.imwrite(save_path.replace(path.suffix, '_heat_'+path.suffix),
                        heat_map*255)
            LOGGER.info(f'saved image to {save_path}')
        else: 
            return cam_image

def parseopt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--method',
                        type=str,
                        default='EigenCAM',
                        help='the method to use for interpreting the feature maps')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')

    parser.add_argument('--verbose', action='store_true', help='verbose log')
    parser.add_argument('--layer', type=int, default=-2, help="layer to backpropagate gradients to")
    parser.add_argument('--class-names', nargs='*',default='', help='filter by class: --classes dog, or --classes dog cat')

    parser.add_argument('--keep-only-topk', type=int, default=100, help="percentage of heatmap pixels to keep")
    parser.add_argument('--backprop-array', nargs='*',default='', help="backprop array items" )
    parser.add_argument('--backward-per-class', type=bool, default=False, help="whether the method should backprop per each class or do it all at one backward")
    parser.add_argument('--crop', type=bool, default=False, help="use this if you want to crop heatmap area in order to evaluate methods for interpretability")
    

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
