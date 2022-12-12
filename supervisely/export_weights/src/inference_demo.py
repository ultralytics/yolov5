import sys

sys.path.insert(0, './yolov5')

import argparse
import numpy as np

import os
import supervisely as sly
import torch
import torch.nn.functional as F

from pathlib import Path
from torchvision import transforms

from app_utils import to_numpy, get_image, get_model, get_configs, preprocess
from supervisely.serve.src.nn_utils import construct_model_meta
from utils.general import non_max_suppression


# from utils.torch_utils import select_device
# my_app = sly.AppService()
# team_id = int(os.environ['context.teamId'])
# workspace_id = int(os.environ['context.workspaceId'])
# task_id = int(os.environ['TASK_ID'])
# customWeightsPath = os.environ['modal.state.slyFile']
# device = select_device(device='cpu')


def infer_torch_model(_model, _tensor):
    output_ = _model(_tensor)[0]
    return output_


def infer_onnx_model(_model, _tensor):
    _model, input_name, label_name = _model
    onnx_model_inference = _model.run([label_name], {input_name: to_numpy(_tensor).astype(np.float32)})[0]
    return onnx_model_inference


def infer_model(model_, img, conf_thresh=0.25, iou_thresh=0.45,
                agnostic=False, input_img_size=None):
    infer_fn = infer_onnx_model if isinstance(model_, tuple) else infer_torch_model
    if hasattr(model_, 'img_size'):
        height, width = model_.img_size
    else:
        if input_img_size:
            height, width = input_img_size
        else:
            raise Exception('input_image_size must be passed for TorchScript and ONNx models')
    resize = transforms.Resize(size=(height, width))
    prepared_image = resize(img)

    output_ = infer_fn(model_, prepared_image)
    output_ = non_max_suppression(output_,
                                  conf_thres=conf_thresh,
                                  iou_thres=iou_thresh,
                                  agnostic=agnostic)

    original_img_sz = img.shape[2:]
    reshaped_img_sz = prepared_image.shape[2:]
    output_ = preprocess(output_, original_img_sz, reshaped_img_sz)
    return output_


def sliding_window(model_, img, conf_thresh=0.25, iou_thresh=0.45,
                   agnostic=False, native=False,
                   sliding_window_step=[], input_img_size=None):
    infer_fn = infer_onnx_model if isinstance(model_, tuple) else infer_torch_model
    img_h, img_w = img.shape[-2:]
    try:
        sw_h, sw_w = model_.img_size
    except:
        assert input_img_size is not None, 'For torchScript and ONNX models input image size should be passed!'
        sw_h, sw_w = input_img_size

    if sliding_window_step:
        sws_h, sws_w = sliding_window_step
    else:
        sws_h = round((img_h - sw_h + 1) / 5)
        sws_w = round((img_w - sw_w + 1) / 5)

    possible_height_steps = (img_h - sw_h + 1) // sws_h
    possible_width_steps = (img_w - sw_w + 1) // sws_w

    candidates = []

    for w in range(possible_width_steps + 1):
        for h in range(possible_height_steps + 1):
            top = h * sws_h
            left = w * sws_w
            bot = top + sw_h
            right = left + sw_w
            cropped_image = img[..., top:bot, left:right]
            # padding_left, padding_right, padding_top, padding_bottom
            cropped_image = F.pad(cropped_image,
                                  pad=(0, sw_w - cropped_image.shape[3],
                                       0, sw_h - cropped_image.shape[2]))
            inf_res = infer_fn(model_, cropped_image)
            inf_res = inf_res[inf_res[..., 4] > conf_thresh]
            inf_res[:, 0] += left
            inf_res[:, 1] += top
            if native:
                inf_res = inf_res if len(inf_res.shape) == 3 else np.expand_dims(inf_res, axis=0)
                inf_res = non_max_suppression(inf_res,
                                              conf_thres=conf_thresh,
                                              iou_thres=iou_thresh,
                                              agnostic=agnostic)[0]
            candidates.append(inf_res)

    if isinstance(candidates[0], np.ndarray):
        candidates = [torch.as_tensor(element) for element in candidates]
    detections = torch.cat(candidates).unsqueeze_(0)

    if not native:
        detections = non_max_suppression(detections, conf_thres=conf_thresh, iou_thres=iou_thresh, agnostic=agnostic)
    return detections


def visualize_dets(img_, output_, save_path_, names_, meta_):
    labels = []
    for i, det in enumerate(output_):
        if det is not None and len(det):
            for *xyxy, conf, cls in reversed(det):
                left, top, right, bottom = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                rect = sly.Rectangle(top, left, bottom, right)
                obj_class = meta_.get_obj_class(names_[int(cls)])
                tag = sly.Tag(meta_.get_tag_meta("confidence"), round(float(conf), 4))
                label = sly.Label(rect, obj_class, sly.TagCollection([tag]))
                labels.append(label)

    width, height = img_.size
    ann = sly.Annotation(img_size=(height, width), labels=labels)

    vis = np.copy(img_)
    ann.draw_contour(vis, thickness=2)
    sly.image.write(save_path_, vis)
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        # default='/home/work/PycharmProjects/app_debug_data/data/best.onnx',
                        help='initial weights path')
    parser.add_argument('--cfgs', type=str,
                        # default='/home/work/PycharmProjects/app_debug_data/data/opt.yaml',
                        help='path to model cfgs (required for ONNX anf TorchScript models)')
    parser.add_argument('--image', type=str,
                        # default='./IMG_0748_big.jpeg',
                        help='initial image path')
    parser.add_argument('--mode',
                        choices=['direct', 'sliding_window'],
                        default='direct',
                        help='inference mode')
    parser.add_argument('--conf_thresh', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou_thresh', type=float, default=0.45, help='Intersection over Union threshold')
    parser.add_argument('--agnostic', type=bool, default=False, help='')
    parser.add_argument('--native', type=bool, default=False, help='for sliding window approach')
    parser.add_argument('--sliding_window_step', nargs='+', type=int,
                        # default=[1216, 1216],
                        help='')
    parser.add_argument('--viz', type=bool, default=False, help='to save detections to image')
    parser.add_argument('--original_model',
                        # default='/home/work/PycharmProjects/app_debug_data/data/best.pt',
                        help='path to original model to construct meta (required for ONNX anf TorchScript models)')
    parser.add_argument('--save_path', type=str,
                        default=os.path.join(os.getcwd(), "vis_detection.jpg"),
                        help='path to save inference results')

    opt = parser.parse_args()
    folder = Path(opt.weights).parents[0]
    file = Path(opt.weights).name.split('.')[0] + '.pt'
    path2original_model = os.path.join(folder, file)
    if opt.viz and not os.path.exists(path2original_model):
        print('Please, set path to original_model to construct meta to visualise results.')
        raise FileNotFoundError(path2original_model)

    cfgs = get_configs(opt.cfgs)
    input_img_size = cfgs['img_size']

    # load and prepare image
    tensor, image = get_image(opt.image)
    # load and prepare models
    model = get_model(opt.weights)

    # infer prepared image
    if opt.mode == 'direct':
        output = infer_model(model_=model, img=tensor, conf_thresh=opt.conf_thresh, iou_thresh=opt.iou_thresh,
                             agnostic=opt.agnostic, input_img_size=input_img_size)
    else:
        output = sliding_window(model_=model, img=tensor, conf_thresh=opt.conf_thresh, iou_thresh=opt.iou_thresh,
                                agnostic=opt.agnostic, native=opt.native, sliding_window_step=opt.sliding_window_step,
                                input_img_size=input_img_size)

    print(opt.viz)
    if opt.viz:
        # load orig YOLOv5 model to construct meta
        o_model = get_model(opt.original_model)
        # meta construction
        meta = construct_model_meta(o_model)
        # get class names
        names = o_model.module.names if hasattr(o_model, 'module') else o_model.names
        visualize_dets(image, output, opt.save_path, names, meta)


if __name__ == '__main__':
    main()
