import numpy as np
import onnxruntime as rt
import os
import supervisely_lib as sly
import torch
import yaml

from pathlib import Path
from PIL import Image
from torchvision import transforms
from app_utils import download_weights, to_numpy
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from supervisely.serve.src.nn_utils import construct_model_meta


my_app = sly.AppService()
team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
task_id = int(os.environ['TASK_ID'])
customWeightsPath = os.environ['modal.state.slyFile']
device = select_device(device='cpu')

kwargs = dict(my_app=my_app,
            team_id=team_id,
            workspace_id=workspace_id,
            task_id=task_id,
            customWeightsPath=customWeightsPath,
            device=device)


def onnx_inference(path_to_onnx_saved_model):
    onnx_model = rt.InferenceSession(path_to_onnx_saved_model)
    input_name = onnx_model.get_inputs()[0].name
    label_name = onnx_model.get_outputs()[0].name
    return onnx_model, input_name, label_name


def sliding_window_approach(model_, image, **kwargs):
    conf_threshold = kwargs['conf_threshold'] if 'conf_threshold' in kwargs else 0.25
    iou_threshold = kwargs['iou_threshold'] if 'iou_threshold' in kwargs else 0.45
    agnostic = kwargs['agnostic'] if 'agnostic' in kwargs else False
    native = kwargs['native'] if 'native' in kwargs else True
    sliding_window_step = kwargs['sliding_window_step'] if 'sliding_window_step' in kwargs else None
    input_iamge_size = kwargs['input_iamge_size'] if 'input_iamge_size' in kwargs else None

    if isinstance(model_, tuple):
        onnx_model, input_name, label_name = model_
    img_h, img_w = image.shape[-2:]
    try:
        sw_h, sw_w = model_.img_size
    except:
        assert input_iamge_size is not None, 'For torchScript and ONNX models input image size should be passed!'
        sw_h, sw_w = input_iamge_size

    if sliding_window_step:
        sws_h, sws_w = sliding_window_step
    else:
        sws_h = (img_h - sw_h + 1) // 4
        sws_w = (img_w - sw_w + 1) // 4

    possible_height_steps = (img_h - sw_h + 1) // sws_h
    possible_width_steps = (img_w - sw_w + 1) // sws_w

    candidates = []

    for w in range(possible_width_steps + 1):
        for h in range(possible_height_steps + 1):
            top = h * sws_h
            left = w * sws_w
            bot = top + sw_h
            right = left + sw_w
            cropped_image = image[..., top:bot, left:right].unsqueeze(0) / 255
            if not isinstance(model_, tuple):
                inf_res = model_(cropped_image)[0]
            else:
                inf_res = onnx_model.run([label_name], {input_name: to_numpy(cropped_image).astype(np.float32)})[0]

            inf_res = inf_res[inf_res[..., 4] > conf_threshold]
            inf_res[:, 0] += left
            inf_res[:, 1] += top
            if native:
                inf_res = inf_res if len(inf_res.shape) == 3 else np.expand_dims(inf_res, axis=0)
                inf_res = non_max_suppression(inf_res,
                                              conf_thres=conf_threshold,
                                              iou_thres=iou_threshold,
                                              agnostic=agnostic)[0]
            candidates.append(inf_res)

    if isinstance(candidates[0], np.ndarray):
        candidates = [torch.as_tensor(element) for element in candidates]  # if not isinstance(element, torch.Tensor)
    detections = torch.cat(candidates).unsqueeze_(0)

    if not native:
        detections = non_max_suppression(detections, conf_thres=conf_threshold, iou_thres=iou_threshold,
                                         agnostic=agnostic)
    return detections


def visualize_dets(img0, output, save_path, **kwargs):
    labels = []
    names = kwargs['model'].module.names if hasattr(model, 'module') else kwargs['model'].names
    for i, det in enumerate(output):  # replace output with "torchScript_output" or "onnx_output"
        if det is not None and len(det):
            for *xyxy, conf, cls in reversed(det):
                left, top, right, bottom = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                rect = sly.Rectangle(top, left, bottom, right)
                obj_class = meta.get_obj_class(names[int(cls)])
                tag = sly.Tag(meta.get_tag_meta("confidence"), round(float(conf), 4))
                label = sly.Label(rect, obj_class, sly.TagCollection([tag]))
                labels.append(label)

    # height, width = img0.shape[2:]
    width, height = img0.size
    ann = sly.Annotation(img_size=(height, width), labels=labels)

    vis = np.copy(img0)
    ann.draw_contour(vis, thickness=2)
    sly.image.write(os.path.join(save_path, "vis_detection.jpg"), vis)
    return vis


def prepare_model(weights_path, **kwargs):
    cwp = os.path.join(Path(weights_path).parents[1], 'opt.yaml')
    cfgs_path = download_weights(cwp, **kwargs)
    kwargs['configs_path'] = cfgs_path
    path_to_saved_model = download_weights(weights_path, **kwargs)

    if 'pt' in weights_path:
        if 'torchscript' in weights_path:
            model = torch.jit.load(path_to_saved_model)
            return model
        else:
            model = attempt_load(weights=path_to_saved_model)  # , map_location=device
            kwargs['model'] = model
            return model, kwargs
    if 'onnx' in weights_path:
        model = onnx_inference(path_to_saved_model)
        return model


def infer_torch_model(torch_script_model, tensor):
    # simple inference for torchScript:
    torch_script_model_inference = torch_script_model(tensor)[0]
    return torch_script_model_inference


def infer_onnx_model(onnx_model, tensor):
    # simple inference for ONNX:
    onnx_model, input_name, label_name = onnx_model
    onnx_model_inference = onnx_model.run([label_name], {input_name: to_numpy(tensor).astype(np.float32)})[0]
    return onnx_model_inference


def infer_model(model_, image, simple_inference=True, **kwargs):
    if simple_inference:
        infer_fn = infer_onnx_model if isinstance(model, tuple) else infer_torch_model
        if len(image.shape) == 3:
            slice = image.unsqueeze(0)
        if slice.max() > 1:
            slice = slice / 255
        height, width = kwargs['input_iamge_size']
        if slice.shape[2] > height or slice.shape[3] > width:
            slice = slice[..., :height, :width]

        model_inference = infer_fn(model_, slice)
        output = non_max_suppression(model_inference,
                                     conf_thres=kwargs['conf_threshold'],
                                     iou_thres=kwargs['iou_threshold'],
                                     agnostic=kwargs['agnostic'])
    else:
        output = sliding_window_approach(model, image, **kwargs)
    return output


customWeightsPath = '/yolov5_train/Lemons-aug/8342_008/weights/best.pt'
customWeightsPath_torchScript = '/yolov5_train/Lemons-aug/8342_008/weights/best.torchscript.pt'
customWeightsPath_onnx = "/yolov5_train/Lemons-aug/8342_008/weights/best.onnx"

# models init stage
model, kwargs = prepare_model(customWeightsPath, **kwargs)
torch_script_model = prepare_model(customWeightsPath_torchScript, **kwargs)
onnx_model = prepare_model(customWeightsPath_onnx, **kwargs)

# metadata construction stage
meta = construct_model_meta(model)

# download target image for inference
path_to_image = './IMG_0748_big.jpeg'
big_image = Image.open(path_to_image)
tensor = transforms.PILToTensor()(big_image)
image = transforms.ToPILImage()(tensor)

# set additional cfgs
try:
    H, W = model.img_size
    kwargs['input_iamge_size'] = [H, W]
except:
    with open(kwargs['configs_path'], 'r') as yaml_file:
        cfgs = yaml.load(yaml_file)
    input_iamge_size = cfgs['img_size']  # custom model input image size

kwargs['conf_threshold'] = 0.25
kwargs['iou_threshold'] = 0.45
kwargs['agnostic'] = False

rez = infer_model(onnx_model, tensor, simple_inference=False, **kwargs)
visualize_dets(img0=image, output=rez, save_path=os.getcwd(), **kwargs)
