import os
import json
import yaml
import pathlib
import sys
import supervisely_lib as sly

root_source_path = str(pathlib.Path(sys.argv[0]).parents[3])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)

from nn_utils import construct_model_meta, load_model, inference

my_app = sly.AppService()

TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])

meta: sly.ProjectMeta = None

modelWeightsOptions = os.environ['modal.state.modelWeightsOptions']
pretrained_weights = os.environ['modal.state.modelSize']
custom_weights = os.environ['modal.state.weightsPath']


DEVICE_STR = os.environ['modal.state.device']
final_weights = None
model = None
half = None
device = None
imgsz = None


settings_path = os.path.join(root_source_path, "supervisely/serve/custom_settings.yaml")
sly.logger.info(f"Custom inference settings path: {settings_path}")
with open(settings_path, 'r') as file:
    default_settings_str = file.read()
    default_settings = yaml.safe_load(default_settings_str)


@my_app.callback("get_output_classes_and_tags")
@sly.timeit
def get_output_classes_and_tags(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    my_app.send_response(request_id, data=meta.to_json())


@my_app.callback("get_session_info")
@sly.timeit
def get_session_info(api: sly.Api, task_id, context, state, app_logger):
    info = {
        "app": "YOLOv5 serve",
        "weights": final_weights,
        "device": str(device),
        "half": str(half),
        "input_size": imgsz,
        "session_id": task_id,
        "classes_count": len(meta.obj_classes),
        "tags_count": len(meta.tag_metas),
    }
    request_id = context["request_id"]
    my_app.send_response(request_id, data=info)


@my_app.callback("get_custom_inference_settings")
@sly.timeit
def get_custom_inference_settings(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    my_app.send_response(request_id, data={"settings": default_settings_str})


def inference_image_path(image_path, context, state, app_logger):
    app_logger.debug("Input path", extra={"path": image_path})
    settings = state["settings"]

    rect = None
    if "rectangle" in state:
        top, left, bottom, right = state["rectangle"]
        rect = sly.Rectangle(top, left, bottom, right)
    for key, value in default_settings.items():
        if key not in settings:
            app_logger.warn("Field {!r} not found in inference settings. Use default value {!r}".format(key, value))
    debug_visualization = settings.get("debug_visualization", default_settings["debug_visualization"])
    conf_thres = settings.get("conf_thres", default_settings["conf_thres"])
    iou_thres = settings.get("iou_thres", default_settings["iou_thres"])
    augment = settings.get("augment", default_settings["augment"])

    image = sly.image.read(image_path)  # RGB image
    if rect is not None:
        image = sly.image.crop(image, rect)
    ann_json = inference(model, half, device, imgsz, image, meta,
                         conf_thres=conf_thres, iou_thres=iou_thres, augment=augment,
                         debug_visualization=debug_visualization)
    return ann_json


@my_app.callback("inference_image_url")
@sly.timeit
def inference_image_url(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})

    image_url = state["image_url"]
    ext = sly.fs.get_file_ext(image_url)
    if ext == "":
        ext = ".jpg"
    local_image_path = os.path.join(my_app.data_dir, sly.rand_str(15) + ext)

    sly.fs.download(image_url, local_image_path)
    ann_json = inference_image_path(local_image_path, context, state, app_logger)
    sly.fs.silent_remove(local_image_path)

    request_id = context["request_id"]
    my_app.send_response(request_id, data=ann_json)


@my_app.callback("inference_image_id")
@sly.timeit
def inference_image_id(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_id = state["image_id"]
    image_info = api.image.get_info_by_id(image_id)
    state["image_url"] = image_info.full_storage_url
    inference_image_url(api, task_id, context, state, app_logger)


@my_app.callback("inference_batch_ids")
@sly.timeit
def inference_batch_ids(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    ids = state["batch_ids"]
    infos = api.image.get_info_by_id_batch(ids)
    paths = []
    for info in infos:
        paths.append(os.path.join(my_app.data_dir, sly.rand_str(10) + info.name))
    api.image.download_paths(infos[0].dataset_id, ids, paths)

    results = []
    for image_path in paths:
        ann_json = inference_image_path(image_path, context, state, app_logger)
        results.append(ann_json)
        sly.fs.silent_remove(image_path)

    request_id = context["request_id"]
    my_app.send_response(request_id, data=results)


def debug_inference():
    image = sly.image.read("./data/images/bus.jpg")  # RGB
    ann = inference(model, half, device, imgsz, image, meta, debug_visualization=True)
    print(json.dumps(ann, indent=4))


@my_app.callback("preprocess")
@sly.timeit
def preprocess(api: sly.Api, task_id, context, state, app_logger):
    global model, half, device, imgsz, meta, final_weights

    # download weights
    progress = sly.Progress("Downloading weights", 1, is_size=True, need_info_log=True)
    local_path = os.path.join(my_app.data_dir, "weights.pt")
    if modelWeightsOptions == "pretrained":
        url = os.path.join("https://github.com/ultralytics/yolov5/releases/download/v4.0/", pretrained_weights)
        final_weights = url
        sly.fs.download(url, local_path, my_app.cache, progress)
    elif modelWeightsOptions == "custom":
        final_weights = custom_weights
        file_info = api.file.get_info_by_path(TEAM_ID, custom_weights)
        progress.set(current=0, total=file_info.sizeb)
        api.file.download(TEAM_ID, custom_weights, local_path, my_app.cache, progress.iters_done_report)
    else:
        raise ValueError("Unknown weights option {!r}".format(modelWeightsOptions))

    # load model on device
    model, half, device, imgsz = load_model(local_path, device=DEVICE_STR)
    meta = construct_model_meta(model)
    sly.logger.info("Model has been successfully deployed")


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": TEAM_ID,
        "context.workspaceId": WORKSPACE_ID,
        "modal.state.modelWeightsOptions": modelWeightsOptions,
        "modal.state.modelSize": pretrained_weights,
        "modal.state.weightsPath": custom_weights
    })

    my_app.run(initial_events=[{"command": "preprocess"}])


#@TODO: augment inference
#@TODO: https://pypi.org/project/cachetools/
if __name__ == "__main__":
    sly.main_wrapper("main", main)