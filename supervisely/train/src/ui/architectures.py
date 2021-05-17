import errno
import os
import sly_train_globals as g
from sly_utils import get_progress_cb
import supervisely_lib as sly


def get_models_list():
    return [
        {
            "config": "",
            "weightsUrl": "",
            "Model": "YOLOv5s",
            "Size": 640,
            "mAP^val": 36.7,
            "mAP^test": 36.7,
            "mAP^val_0.5": 55.4,
            "Speed": 2.0,
            "Params": 7.3,
            "FLOPS": 17.0,
        },
        {
            "config": "",
            "weightsUrl": "",
            "Model": "YOLOv5m",
            "Size": 640,
            "mAP^val": 44.5,
            "mAP^test": 44.5,
            "mAP^val_0.5": 63.1,
            "Speed": 2.7,
            "Params": 21.4,
            "FLOPS": 51.3,
        },
        {
            "config": "",
            "weightsUrl": "",
            "Model": "YOLOv5l",
            "Size": 640,
            "mAP^val": 48.2,
            "mAP^test": 48.2,
            "mAP^val_0.5": 66.9,
            "Speed": 3.8,
            "Params": 47.0,
            "FLOPS": 115.4,
        },
        {
            "config": "",
            "weightsUrl": "",
            "Model": "YOLOv5x",
            "Size": 640,
            "mAP^val": 50.4,
            "mAP^test": 50.4,
            "mAP^val_0.5": 68.8,
            "Speed": 6.1,
            "Params": 87.7,
            "FLOPS": 218.8,
        },
        {
            "config": "",
            "weightsUrl": "",
            "Model": "YOLOv5s6",
            "Size": 1280,
            "mAP^val": 43.3,
            "mAP^test": 43.3,
            "mAP^val_0.5": 61.9,
            "Speed": 4.3,
            "Params": 12.7,
            "FLOPS": 17.4,
        },
        {
            "config": "",
            "weightsUrl": "",
            "Model": "YOLOv5m6",
            "Size": 1280,
            "mAP^val": 50.5,
            "mAP^test": 50.5,
            "mAP^val_0.5": 68.7,
            "Speed": 8.4,
            "Params": 35.9,
            "FLOPS": 52.4,
        },
        {
            "config": "",
            "weightsUrl": "",
            "Model": "YOLOv5l6",
            "Size": 1280,
            "mAP^val": 53.4,
            "mAP^test": 53.4,
            "mAP^val_0.5": 71.1,
            "Speed": 12.3,
            "Params": 77.2,
            "FLOPS": 117.7,
        },
        {
            "config": "",
            "weightsUrl": "",
            "Model": "YOLOv5x6",
            "Size": 1280,
            "mAP^val": 54.4,
            "mAP^test": 54.4,
            "mAP^val_0.5": 72.0,
            "Speed": 22.4,
            "Params": 141.8,
            "FLOPS": 222.9,
        },
    ]


def get_table_columns():
    return [
        {"key": "Model", "title": "Model", "subtitle": None},
        {"key": "Size", "title": "Size", "subtitle": "(pixels)"},
        {"key": "mAP^val", "title": "mAP<sub>val</sub>", "subtitle": "0.5:0.95"},
        {"key": "mAP^test", "title": "mAP<sub>test</sub>", "subtitle": "0.5:0.95"},
        {"key": "mAP^val_0.5", "title": "mAP<sub>val</sub>", "subtitle": "0.5"},
        {"key": "Speed", "title": "Speed", "subtitle": "V100 (ms)"},
        {"key": "Params", "title": "Params", "subtitle": "(M)"},
        {"key": "FLOPS", "title": "FLOPS", "subtitle": "640 (B)"},
    ]


def init(data, state):
    data["models"] = get_models_list()
    data["modelColumns"] = get_table_columns()
    state["selectedModel"] = "YOLOv5s"
    state["weightsInitialization"] = "coco"

    # @TODO: for debug
    #state["weightsPath"] = "/yolov5_train/coco128_002/2390/weights/best.pt"
    state["weightsPath"] = ""


def prepare_weights(state):
    if state["weightsInitialization"] == "custom":
        # download custom weights
        weights_path_remote = state["weightsPath"]
        if not weights_path_remote.endswith(".pt"):
            raise ValueError(f"Weights file has unsupported extension {sly.fs.get_file_ext(weights_path_remote)}. "
                             f"Supported: '.pt'")
        weights_path_local = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_path_remote))
        file_info = g.api.file.get_info_by_path(g.team_id, weights_path_remote)
        if file_info is None:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), weights_path_remote)
        progress_cb = get_progress_cb("Download weights", file_info.sizeb, is_size=True)
        g.api.file.download(g.team_id, weights_path_remote, weights_path_local, g.my_app.cache, progress_cb)

        state["_weightsPath"] = weights_path_remote
        state["weightsPath"] = weights_path_local
    else:
        model_name = state['selectedModel'].lower()
        state["weightsPath"] = f"{model_name}.pt"
        sly.logger.info("Pretrained COCO weights will be added automatically")
