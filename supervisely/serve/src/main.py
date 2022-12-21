import os
import sys
try:
    from typing import Literal
except:
    from typing_extensions import Literal
from typing import List, Dict, Any
import yaml
from dotenv import load_dotenv
import torch
import numpy as np
import supervisely as sly
from supervisely.geometry.sliding_windows_fuzzy import SlidingWindowsFuzzy
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xywh2xyxy
from utils.datasets import letterbox
from pathlib import Path

root_source_path = str(Path(__file__).parents[3])
app_source_path = str(Path(__file__).parents[1])
load_dotenv(os.path.join(app_source_path, "local.env"))
load_dotenv(os.path.expanduser("~/supervisely.env"))

model_weights_options = os.environ['modal.state.modelWeightsOptions']
pretrained_weights = os.environ['modal.state.selectedModel'].lower()
custom_weights = os.environ['modal.state.weightsPath']

class YOLOv5Model(sly.nn.inference.ObjectDetection):
    def load_on_device(
        self,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        # download weights
        if model_weights_options == "pretrained":
            self.local_weights_path = self.location
        if model_weights_options == "custom":
            self.local_weights_path = self.location[0]
            configs_local_path = self.location[1]

        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = attempt_load(self.local_weights_path, map_location=device)  # load FP32 model
        try:
            with open(configs_local_path, 'r') as stream:
                cfgs_loaded = yaml.safe_load(stream)
        except:
            cfgs_loaded = None

        if hasattr(self.model, 'module') and hasattr(self.model.module, 'img_size'):
            imgsz = self.model.module.img_size[0]
        elif hasattr(self.model, 'img_size'):
            imgsz = self.model.img_size[0]
        elif cfgs_loaded is not None and cfgs_loaded['img_size']:
            imgsz = cfgs_loaded['img_size'][0]
        else:
            default_img_size = 640
            sly.logger.warning(f"Image size is not found in model checkpoint. Use default: {default_img_size}")
            imgsz = default_img_size
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size

        if self.half:
            self.model.half()  # to FP16

        if self.device.type != 'cpu':
            self.model(
                torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters()))
            )  # run once

        self.class_names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        colors = None
        if hasattr(self.model, 'module') and hasattr(self.model.module, 'colors'):
            colors = self.model.module.colors
        elif hasattr(self.model, 'colors'):
            colors = self.model.colors
        else:
            colors = []
            for i in range(len(self.class_names)):
                colors.append(sly.color.generate_rgb(exist_colors=colors))

        obj_classes = [sly.ObjClass(name, sly.Rectangle, color) for name, color in zip(self.class_names, colors)]

        self._model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes),
                                tag_metas=sly.TagMetaCollection([self._get_confidence_tag_meta()]))

        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def get_classes(self) -> List[str]:
        return self.class_names  # e.g. ["cat", "dog", ...]

    def get_info(self):
        info = super().get_info()
        info["model_name"] = "YOLOv5"
        info["checkpoint_name"] = pretrained_weights
        info["pretrained_on_dataset"] = "COCO train 2017" if model_weights_options == "pretrained" else "custom"
        info["device"] = self.device.type
        info["sliding_window_support"] = self.sliding_window_mode
        info["half"] = str(self.half)
        info["input_size"] = self.imgsz
        return info

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[sly.nn.PredictionBBox]:
        conf_thres = settings.get("conf_thres", self.custom_inference_settings["conf_thres"])
        iou_thres = settings.get("iou_thres", self.custom_inference_settings["iou_thres"])

        augment = settings.get("augment", self.custom_inference_settings["augment"])
        # inference_mode = settings.get("inference_mode", "full")
        image = sly.image.read(image_path)  # RGB image
        predictions = []

        img0 = image
        # Padded resize
        img = letterbox(img0, new_shape=self.imgsz, stride=self.stride)[0]
        img = img.transpose(2, 0, 1)  # to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        inf_out = self.model(img, augment=augment)[0]

        # Apply NMS
        output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, agnostic=False)

        for det in output:
            if det is not None and len(det) > 0:
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    bbox = [int(xyxy[1]), int(xyxy[0]), int(xyxy[3]), int(xyxy[2])]
                    predictions.append(sly.nn.PredictionBBox(self.class_names[int(cls)], bbox, conf.item()))

        return predictions


    def predict_raw(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[sly.nn.PredictionBBox]:
        conf_thres = settings.get("conf_thres")

        augment = settings.get("augment")
        # inference_mode = settings.get("inference_mode", "full")
        image = sly.image.read(image_path)  # RGB image
        predictions = []

        img0 = image
        # Padded resize
        img = letterbox(img0, new_shape=self.imgsz, stride=self.stride)[0]
        img = img.transpose(2, 0, 1)  # to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        inf_out = self.model(img, augment=augment)[0][0]
        
        inf_out[:, 5:] *= inf_out[:, 4:5]  # conf = obj_conf * cls_conf
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(inf_out[:, :4])
        conf, j = inf_out[:, 5:].max(1, keepdim=True) # best class
        det = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        for *xyxy, conf, cls in reversed(det):
            bbox = [int(xyxy[1]), int(xyxy[0]), int(xyxy[3]), int(xyxy[2])]
            predictions.append(sly.nn.PredictionBBox(self.class_names[int(cls)], bbox, conf.item()))

        return predictions

sly.logger.info("Script arguments", extra={
    "teamId": sly.env.team_id(),
    "workspaceId": sly.env.workspace_id(),
    "modal.state.modelWeightsOptions": model_weights_options,
    "modal.state.modelSize": pretrained_weights,
    "modal.state.weightsPath": custom_weights
})

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

if model_weights_options == "pretrained":
    location = f"https://github.com/ultralytics/yolov5/releases/download/v5.0/{pretrained_weights}.pt"
elif model_weights_options == "custom":
    location = [
        custom_weights, # model weights
        os.path.join(Path(custom_weights).parents[1], 'opt.yaml') # model config
    ]

m = YOLOv5Model(
    location=location, 
    custom_inference_settings=os.path.join(app_source_path, "custom_settings.yaml"),
    sliding_window_mode = "advanced"
)
m.load_on_device(device)

if sly.is_production():
    # this code block is running on Supervisely platform in production
    # just ignore it during development
    m.serve()
else:
    # for local development and debugging
    image_path = "./data/images/bus.jpg"
    settings = {}
    results = m.predict(image_path, settings)
    vis_path = "./data/images/bus_prediction.jpg"
    m.visualize(results, image_path, vis_path)
    print(f"predictions and visualization have been saved: {vis_path}")
