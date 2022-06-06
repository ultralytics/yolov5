import os
from pathlib import Path

import supervisely as sly
import supervisely_app.serve.src.nn_utils as nn_utils

import supervisely_app.serve.src.sly_globals as g


@sly.process_image_roi
def inference_image_path(image_path, project_meta, context, state, app_logger):
    app_logger.debug("Input path", extra={"path": image_path})

    settings = state.get("settings", {})
    for key, value in g.default_settings.items():
        if key not in settings:
            app_logger.warn("Field {!r} not found in inference settings. Use default value {!r}".format(key, value))
    debug_visualization = settings.get("debug_visualization", g.default_settings["debug_visualization"])
    conf_thres = settings.get("conf_thres", g.default_settings["conf_thres"])
    iou_thres = settings.get("iou_thres", g.default_settings["iou_thres"])
    augment = settings.get("augment", g.default_settings["augment"])
    inference_mode = settings.get("inference_mode", "full")

    image = sly.image.read(image_path)  # RGB image
    if inference_mode == "sliding_window":
        ann_json, slides_for_vis = nn_utils.sliding_window_inference(g.model, g.half, g.device, g.imgsz, g.stride,
                                                                     image, g.meta,
                                                                     settings["sliding_window_params"],
                                                                     conf_thres=conf_thres,
                                                                     iou_thres=iou_thres)
        return {"annotation": ann_json, "data": {"slides": slides_for_vis}}
    else:
        ann_json = nn_utils.inference(g.model, g.half, g.device, g.imgsz, g.stride, image, g.meta,
                                      conf_thres=conf_thres, iou_thres=iou_thres, augment=augment,
                                      debug_visualization=debug_visualization)
        return ann_json


# @my_app.callback("preprocess")
@sly.timeit
def preprocess():
    # download weights
    progress = sly.Progress("Downloading weights", 1, is_size=True, need_info_log=True)
    local_path = os.path.join(g.my_app.data_dir, "weights.pt")
    if g.modelWeightsOptions == "pretrained":
        url = f"https://github.com/ultralytics/yolov5/releases/download/v5.0/{g.pretrained_weights}.pt"
        g.final_weights = url
        sly.fs.download(url, local_path, g.my_app.cache, progress)
    elif g.modelWeightsOptions == "custom":
        g.final_weights = g.custom_weights
        configs = os.path.join(Path(g.custom_weights).parents[1], 'opt.yaml')
        configs_local_path = os.path.join(g.my_app.data_dir, 'opt.yaml')
        file_info = g.my_app.public_api.file.get_info_by_path(g.TEAM_ID, g.custom_weights)
        progress.set(current=0, total=file_info.sizeb)
        g.my_app.public_api.file.download(g.TEAM_ID, g.custom_weights, local_path, g.my_app.cache,
                                          progress.iters_done_report)
        g.my_app.public_api.file.download(g.TEAM_ID, configs, configs_local_path)
    else:
        raise ValueError("Unknown weights option {!r}".format(g.modelWeightsOptions))

    # load model on device
    g.model, g.half, g.device, g.imgsz, g.stride = nn_utils.load_model(local_path, device=g.DEVICE_STR)
    g.meta = nn_utils.construct_model_meta(g.model)


def inference_images_dir(img_paths, context, state, app_logger):
    annotations = []
    for image_path in img_paths:
        ann_json = inference_image_path(image_path=image_path,
                                        project_meta=g.meta,
                                        context=context,
                                        state=state,
                                        app_logger=app_logger)
        annotations.append(ann_json)
        sly.fs.silent_remove(image_path)
    return annotations
