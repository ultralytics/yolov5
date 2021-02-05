import os
import json
import yaml
import supervisely_lib as sly

from sly_serve_utils import construct_model_meta, load_model, inference
from supervisely.train.src.sly_utils import load_file_as_string as load_hyp

my_app = sly.AppService()

TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
image_id = 725268

meta: sly.ProjectMeta = None
REMOTE_PATH = os.environ['modal.state.slyFile']

DEVICE_STR = "cpu"
model = None
half = None
device = None
imgsz = None
default_settings = yaml.safe_load(load_hyp("supervisely/serve/custom_settings.yaml"))


@my_app.callback("get_output_classes_and_tags")
@sly.timeit
def get_output_classes_and_tags(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    my_app.send_response(request_id, data=meta.to_json())


@my_app.callback("get_session_info")
@sly.timeit
def get_session_info(api: sly.Api, task_id, context, state, app_logger):
    info = {
        "app": "YOLO v5 serve",
        "weights": REMOTE_PATH,
        "device": str(device),
        "half": str(half),
        "input_size": imgsz
    }
    request_id = context["request_id"]
    my_app.send_response(request_id, data=info)


@my_app.callback("get_custom_inference_settings")
@sly.timeit
def get_custom_inference_settings(api: sly.Api, task_id, context, state, app_logger):
    settings = load_hyp("supervisely/serve/custom_settings.yaml")
    request_id = context["request_id"]
    my_app.send_response(request_id, data={"settings": settings})


@my_app.callback("inference_image_id")
@sly.timeit
def inference_image_id(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_id = state["image_id"]
    settings = state["settings"]

    for key, value in default_settings.items():
        if key not in settings:
            app_logger.warn("Field {!r} not found in inference settings. Use default value {!r}".format(key, value))

    debug_visualization = settings.get("debug_visualization", default_settings["debug_visualization"])
    conf_thres = settings.get("conf_thres", default_settings["conf_thres"])
    iou_thres = settings.get("iou_thres", default_settings["iou_thres"])
    augment = settings.get("augment", default_settings["augment"])

    image = api.image.download_np(image_id)  # RGB image
    ann_json = inference(model, half, device, imgsz, image, meta,
                         conf_thres=conf_thres, iou_thres=iou_thres, augment=augment,
                         debug_visualization=debug_visualization)

    request_id = context["request_id"]
    my_app.send_response(request_id, data=ann_json)


def debug_inference():
    image = sly.image.read("./data/images/bus.jpg")  # RGB
    ann = inference(model, half, device, imgsz, image, meta, debug_visualization=True)
    print(json.dumps(ann, indent=4))


@my_app.callback("preprocess")
@sly.timeit
def preprocess(api: sly.Api, task_id, context, state, app_logger):
    global model, half, device, imgsz, meta

    # download weights
    local_path = os.path.join(my_app.data_dir, sly.fs.get_file_name_with_ext(REMOTE_PATH))
    api.file.download(TEAM_ID, REMOTE_PATH, local_path)

    # load model on device
    model, half, device, imgsz = load_model(local_path, device=DEVICE_STR)
    meta = construct_model_meta(model)


def main():
    my_app.run(initial_events=[{"command": "preprocess"}])


#@TODO: add pretrained models
#@TODO: download progress bar
#@TODO: add arguments to labeling inference (make a fork from NN labeling app)
#@TODO: augment before deploy
#@TODO: log input arguments
#@TODO: fix serve template - debug_inference
#@TODO: deploy on custom device: cpu/gpu
if __name__ == "__main__":
    sly.main_wrapper("main", main)