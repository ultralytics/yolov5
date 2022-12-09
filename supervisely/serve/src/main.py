import os
import sys
try:
    from typing import Literal
except:
    from typing_extensions import Literal
from typing import List, Dict, Any
import cv2
import yaml
import json
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

# from src.demo_data import prepare_weights

root_source_path = str(Path(sys.argv[0]).parents[3])
load_dotenv(os.path.join(root_source_path, "supervisely", "serve", "local.env"))
load_dotenv(os.path.expanduser("~/supervisely.env"))
# prepare_weights()  # prepare demo data automatically for convenient debug

model_weights_options = os.environ['modal.state.modelWeightsOptions']
pretrained_weights = os.environ['modal.state.selectedModel'].lower()
custom_weights = os.environ['modal.state.weightsPath']

IMG_SIZE = 640

class YOLOv5Model(sly.nn.inference.ObjectDetection):
    def load_on_device(
        self,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        # download weights
        progress = sly.Progress("Downloading weights", 1, is_size=True, need_info_log=True)
        self.local_weights_path = os.path.join(sly.app.get_data_dir(), "weights.pt")
        if model_weights_options == "pretrained":
            url = f"https://github.com/ultralytics/yolov5/releases/download/v5.0/{pretrained_weights}.pt"
            sly.fs.download(url, self.local_weights_path, progress=progress)
        elif model_weights_options == "custom":
            configs = os.path.join(Path(custom_weights).parents[1], 'opt.yaml')
            configs_local_path = os.path.join(sly.app.get_data_dir(), 'opt.yaml')
            file_info = self.api.file.get_info_by_path(sly.env.team_id(), custom_weights)
            progress.set(current=0, total=file_info.sizeb)
            self.api.file.download(sly.env.team_id(), custom_weights, self.local_weights_path,
                                            progress_cb=progress.iters_done_report)
            self.api.file.download(sly.env.team_id(), configs, configs_local_path)
        else:
            raise ValueError("Unknown weights option {!r}".format(model_weights_options))

        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = attempt_load(self.local_weights_path, map_location=device)  # load FP32 model
        try:
            configs_path = os.path.join(Path(self.local_weights_path).parents[0], 'opt.yaml')
            with open(configs_path, 'r') as stream:
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
            sly.logger.warning(f"Image size is not found in model checkpoint. Use default: {IMG_SIZE}")
            imgsz = IMG_SIZE
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size

        if self.half:
            self.model.half()  # to FP16

        if self.device.type != 'cpu':
            self.model(
                torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters()))
            )  # run once

        self.custom_settings_path = os.path.join(root_source_path, "supervisely", "serve", "custom_settings.yaml")
        self.class_names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        # self.model_meta = # TODO: can colors exist in model?

        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def get_classes(self) -> List[str]:
        return self.class_names  # e.g. ["cat", "dog", ...]

    def get_info(self):
        info = {
            "app": "YOLOv5 serve",
            "type": "Object Detection",
            "weights": pretrained_weights,
            "device": str(self.device),
            "half": str(self.half),
            "input_size": self.imgsz,
            "session_id": sly.env.task_id(),
            "classes_count": len(self.class_names),
            "sliding_window_support": True,
            "videos_support": True
        }
        return info

    def _get_custom_inference_settings(self) -> str:  # in yaml format
        with open(self.custom_settings_path, 'r') as file:
            default_settings_str = file.read()
        return default_settings_str
    
    def predict_annotation(self, image_path: str, settings: Dict[str, Any]):
        predictions = self.predict(image_path, settings)
        if isinstance(predictions, tuple):
            predictions, slides_for_vis = predictions
            ann = self._predictions_to_annotation(image_path, predictions)
            return {"annotation": ann.to_json(), "data": {"slides": slides_for_vis}}
        return self._predictions_to_annotation(image_path, predictions)

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[sly.nn.PredictionBBox]:
        default_settings = self._get_custom_inference_settings_dict()
        conf_thres = settings.get("conf_thres", default_settings["conf_thres"])
        iou_thres = settings.get("iou_thres", default_settings["iou_thres"])

        augment = settings.get("augment", default_settings["augment"])
        inference_mode = settings.get("inference_mode", "full")
        image = sly.image.read(image_path)  # RGB image
        predictions = []

        if inference_mode == "sliding_window":
            sliding_window_params = settings["sliding_window_params"]
            # 'img' is RGB in [H, W, C] format
            img_h, img_w = image.shape[:2]
            windowHeight = sliding_window_params.get("windowHeight", img_h)
            windowWidth = sliding_window_params.get("windowWidth", img_w)
            overlapY = sliding_window_params.get("overlapY", 0)
            overlapX = sliding_window_params.get("overlapX", 0)
            borderStrategy = sliding_window_params.get("borderStrategy", "shift_window")

            slider = SlidingWindowsFuzzy([windowHeight, windowWidth],
                                        [overlapY, overlapX],
                                        borderStrategy)

            rectangles = []
            for window in slider.get(image.shape[:2]):
                rectangles.append(window)

            candidates = []
            slides_for_vis = []
            frame_size = None
            cropped_img_size = None
            for rect in rectangles:
                cropped_img = image[rect.top:rect.bottom+1, rect.left:rect.right+1]
                if frame_size is None:
                    frame_size = cropped_img.shape[:2]
                cropped_img = letterbox(cropped_img, new_shape=self.imgsz, stride=self.stride)[0]
                if cropped_img_size is None:
                    cropped_img_size = cropped_img.shape

                cropped_img = cropped_img.transpose(2, 0, 1)  # to CxHxW
                cropped_img = np.ascontiguousarray(cropped_img)
                cropped_img = torch.from_numpy(cropped_img).to(self.device)
                cropped_img = cropped_img.half() if self.half else cropped_img.float()  # uint8 to fp16/32
                cropped_img /= 255.0  
                if cropped_img.ndimension() == 3:
                    cropped_img = cropped_img.unsqueeze(0)
                inf_res_base = self.model(cropped_img)[0][0] # inference, out coords xywh
                inf_res_base = inf_res_base[inf_res_base[..., 4] > conf_thres] # remove low box conf (not included class conf)

                if len(inf_res_base) == 0:
                    slides_for_vis.append(None)
                    continue
                inf_res = inf_res_base.clone()
                # prepare dets for vis
                inf_res = xywh2xyxy(inf_res)

                inf_res[:, :4] = scale_coords(cropped_img.shape[-2:], inf_res[:, :4], frame_size).round()
                
                inf_res[:, [0, 2]] += rect.left # x1, x2 to global coords
                inf_res[:, [1, 3]] += rect.top # y1, y2 to global coords
                slides_for_vis.append(inf_res)

                # prepare dets for NMS (to global image coords)
                if len(inf_res_base) > 0:
                    ratio = (cropped_img.shape[-1] / frame_size[1], cropped_img.shape[-2] / frame_size[0])
                    inf_res_base[:, 0] += rect.left * ratio[0]
                    inf_res_base[:, 1] += rect.top * ratio[1]

                candidates.append(inf_res_base)
            
            if isinstance(candidates[0], np.ndarray):
                candidates = [torch.as_tensor(element) for element in candidates]
            detections = torch.cat(candidates).unsqueeze_(0)

            # get raw candidates for vis
            for i, det in enumerate(slides_for_vis):
                if det is None:
                    slides_for_vis[i] = {"rectangle": rectangles[i].to_json(), "labels": []}
                    continue
                labels = []
                for x1, y1, x2, y2, conf, *cls in reversed(det):
                    top, left, bottom, right = y1.int().item(), x1.int().item(), y2.int().item(), x2.int().item()
                    rect = sly.Rectangle(top, left, bottom, right)
                    class_ind = np.argmax(cls).item()
                    max_conf = np.max(cls).item()
                    obj_class = self.model_meta.get_obj_class(self.class_names[class_ind])
                    conf_val = round(conf.float().item(), 4) * max_conf # box_conf * class_conf
                    if conf_val < conf_thres:
                        continue
                    tag = sly.Tag(self._get_confidence_tag_meta(), conf_val)
                    label = sly.Label(rect, obj_class, sly.TagCollection([tag]))
                    labels.append(label.to_json())
                slides_for_vis[i] = {"rectangle": rectangles[i].to_json(), "labels": labels}

            # apply NMS
            detections = non_max_suppression(detections, conf_thres=conf_thres, iou_thres=iou_thres, agnostic=False)
            
            # get labels after NMS
            labels_after_nms = []
            for i, det in enumerate(detections):
                ratio = (image.shape[0] / frame_size[0], image.shape[1] / frame_size[1])
                size = (cropped_img_size[0] * ratio[0], cropped_img_size[1] * ratio[1])
                det[:, :4] = scale_coords(size, det[:, :4], image.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    bbox = [int(xyxy[1]), int(xyxy[0]), int(xyxy[3]), int(xyxy[2])]
                    predictions.append(sly.nn.PredictionBBox(self.class_names[int(cls)], bbox, conf.item()))
                    rect = sly.Rectangle(*bbox)
                    obj_class = self.model_meta.get_obj_class(self.class_names[int(cls)])
                    tag = sly.Tag(self._get_confidence_tag_meta(), round(float(conf), 4))
                    label = sly.Label(rect, obj_class, sly.TagCollection([tag]))
                    labels_after_nms.append(label)
            
            for label_ind, label in enumerate(labels_after_nms):
                labels_after_nms[label_ind] = label.to_json()
            # add two last slides
            full_rect = sly.Rectangle(0, 0, image.shape[0], image.shape[1])
            all_labels_without_nms = []
            for slide in slides_for_vis:
                all_labels_without_nms.extend(slide["labels"])
            slides_for_vis.append({"rectangle": full_rect.to_json(), "labels": all_labels_without_nms})
            slides_for_vis.append({"rectangle": full_rect.to_json(), "labels": labels_after_nms})

            return predictions, slides_for_vis
        else:
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

            for i, det in enumerate(output):
                if det is not None and len(det):
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

# model_dir = sly.env.folder()
# print("Model directory:", model_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

m = YOLOv5Model()
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
