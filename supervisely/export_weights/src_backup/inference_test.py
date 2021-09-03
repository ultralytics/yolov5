import os

from models.experimental import attempt_load
from pathlib import Path
from supervisely_lib.io.fs import  get_file_name_with_ext
from utils.torch_utils import select_device

import onnxruntime as rt
import pathlib
import supervisely_lib as sly
import sys

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from utils.general import non_max_suppression

root_source_path = str(pathlib.Path(sys.argv[0]).parents[3])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)

my_app = sly.AppService()
TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
TASK_ID = int(os.environ['TASK_ID'])
customWeightsPath = os.environ['modal.state.slyFile']
device = select_device(device='cpu')
image_size = 640
ts = None
batch_size = 1
grid = True


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


N = 1  # batch size
C = 3  # number of channels
H = 1216  # 640  # image height
W = 1216  # 640  # image width


def download_weights(path2weights):
    remote_path = path2weights
    weights_path = os.path.join(my_app.data_dir, get_file_name_with_ext(remote_path))
    try:
        my_app.public_api.file.download(team_id=TEAM_ID,
                                        remote_path=remote_path,
                                        local_save_path=weights_path)
        return weights_path
    except:
        raise FileNotFoundError('FileNotFoundError')
        return None


def construct_model_meta(model):
    CONFIDENCE = "confidence"
    names = model.module.names if hasattr(model, 'module') else model.names

    colors = None
    if hasattr(model, 'module') and hasattr(model.module, 'colors'):
        colors = model.module.colors
    elif hasattr(model, 'colors'):
        colors = model.colors
    else:
        colors = []
        for i in range(len(names)):
            colors.append(sly.color.generate_rgb(exist_colors=colors))

    obj_classes = [sly.ObjClass(name, sly.Rectangle, color) for name, color in zip(names, colors)]
    tags = [sly.TagMeta(CONFIDENCE, sly.TagValueType.ANY_NUMBER)]

    meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes),
                           tag_metas=sly.TagMetaCollection(tags))
    return meta


def onnx_inference(path_to_onnx_saved_model):
    onnx_model = rt.InferenceSession(path_to_onnx_saved_model)
    input_name = onnx_model.get_inputs()[0].name
    label_name = onnx_model.get_outputs()[0].name
    return onnx_model, input_name, label_name


def sliding_window_approach(model, image, conf_threshold=0.25, iou_threshold=0.45,
                            agnostic=False, native=False, sliding_window_step: tuple = None,
                            input_iamge_size: tuple = None):
    onnx = False
    if isinstance(model, tuple):
        onnx = True
        onnx_model, input_name, label_name = model
    img_h, img_w = image.shape[-2:]
    try:
        sw_h, sw_w = model.img_size
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
            cropped_image = image[..., top:bot, left:right].unsqueeze(0)/255
            if not onnx:
                inf_res = model(cropped_image)[0]
            else:
                inf_res = onnx_model.run([label_name], {input_name: to_numpy(cropped_image).astype(np.float32)})[0]
            if native:
                inf_res[:, 0] += left
                inf_res[:, 1] += top
                res = non_max_suppression(inf_res, conf_thres=conf_threshold, iou_thres=iou_threshold,
                                          agnostic=agnostic)[0]
                candidates.append(res)
            else:
                inf_res = inf_res[inf_res[..., 4] > conf_threshold]
                inf_res[:, 0] += left
                inf_res[:, 1] += top
                candidates.append(inf_res)

    candidates = [torch.as_tensor(element) for element in candidates if not isinstance(element, torch.Tensor)]
    if native:
        detections = torch.cat(candidates).unsqueeze_(0)
    else:
        candidates = torch.cat(candidates).unsqueeze_(0)
        detections = non_max_suppression(candidates, conf_thres=conf_threshold, iou_thres=iou_threshold, agnostic=agnostic)
    return detections


def visualize_dets(img0, output, save_path):
    labels = []

    names = model.module.names if hasattr(model, 'module') else model.names
    for i, det in enumerate(output):  # replace output with "torchScript_output" or "onnx_output"
        if det is not None and len(det):
            for *xyxy, conf, cls in reversed(det):
                top, left, bottom, right = int(xyxy[1]), int(xyxy[0]), int(xyxy[3]), int(xyxy[2])
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


customWeightsPath = '/yolov5_train/Lemons-aug/8342_008/weights/best.pt'
weights_path = download_weights(customWeightsPath)
configs_path = download_weights(os.path.join(Path(customWeightsPath).parents[1], 'opt.yaml'))
model = attempt_load(weights=weights_path, map_location=device)

H, W = model.img_size
tensor = torch.randn(N, C, H, W)

customWeightsPath_torchScript = '/yolov5_train/Lemons-aug/8342_008/weights/best.torchscript.pt'
# to download converted ONNX weights to local machine
path_to_torch_script_saved_model = download_weights(customWeightsPath_torchScript)
torch_script_model = torch.jit.load(path_to_torch_script_saved_model)
torch_script_model_inference = torch_script_model(tensor)[0]

customWeightsPath_onnx = "/yolov5_train/Lemons-aug/8342_008/weights/best.onnx"
# to download converted ONNX weights to local machine
path_to_onnx_saved_model = download_weights(customWeightsPath_onnx)
onnx_model, input_name, label_name  = onnx_inference(path_to_onnx_saved_model)
onnx_model_inference = onnx_model.run([label_name], {input_name: to_numpy(tensor).astype(np.float32)})[0]

torchScript_output = non_max_suppression(torch_script_model_inference, conf_thres=0.25, iou_thres=0.45, agnostic=False)
onnx_output = non_max_suppression(onnx_model_inference, conf_thres=0.25, iou_thres=0.45, agnostic=False)
meta = construct_model_meta(model)
# ======================================================================================================================
path_to_image = './IMG_0748_big.jpeg'
big_image = Image.open(path_to_image)
tensor = transforms.PILToTensor()(big_image)
image = transforms.ToPILImage()(tensor)

customWeightsPath = '/yolov5_train/Lemons-aug/8342_008/weights/best.pt'
weights_path = download_weights(customWeightsPath)
configs_path = download_weights(os.path.join(Path(customWeightsPath).parents[1], 'opt.yaml'))
model = attempt_load(weights=weights_path, map_location=device)

# customWeightsPath_torchScript = '/yolov5_train/Lemons-aug/8342_008/weights/best.torchscript.pt'
# path_to_torch_script_saved_model = download_weights(customWeightsPath_torchScript)
# torch_script_model = torch.jit.load(path_to_torch_script_saved_model)

# customWeightsPath_onnx = "/yolov5_train/Lemons-aug/8342_008/weights/best.onnx"
# path_to_onnx_saved_model = download_weights(customWeightsPath_onnx)
# onnx_model, input_name, label_name  = onnx_inference(path_to_onnx_saved_model)

# MODEL = model # original model
# MODEL = torch_script_model # converted torchScript model
MODEL = onnx_inference(path_to_onnx_saved_model)

try:
    H, W = model.img_size
    input_iamge_size = [H, W]
except:
    input_iamge_size = [1216, 1216]  # custom model input image size

rez = sliding_window_approach(MODEL, tensor, conf_threshold=0.25, iou_threshold=0.45, input_iamge_size=input_iamge_size)
visualize_dets(img0=image, output=rez, save_path=os.getcwd())

