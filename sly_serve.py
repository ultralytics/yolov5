import os
import numpy as np
import json
import supervisely_lib as sly

from sly_serve_utils import construct_model_meta, load_model, inference

my_app = sly.AppService()

TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
image_id = 725268

meta: sly.ProjectMeta = None
REMOTE_PATH = "/yolov5_train/coco128_002/2278_072/weights/best.pt"
DEVICE_STR = "cpu"
model = None
half = None
device = None
imgsz = None


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
        "model": REMOTE_PATH,
        "device": "???"
    }
    request_id = context["request_id"]
    my_app.send_response(request_id, data=info)


@my_app.callback("inference_image_id")
@sly.timeit
def inference_image_id(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_id = state["image_id"]
    debug_visualization = state.get("debug_visualization", False)
    image = api.image.download_np(image_id)  # RGB image
    ann_json = inference(image, debug_visualization)

    request_id = context["request_id"]
    my_app.send_response(request_id, data=ann_json)


def debug_inference():
    image = sly.image.read("./data/images/bus.jpg")  # RGB
    ann = inference(model, half, device, image, imgsz, meta, debug_visualization=True)
    print(json.dumps(ann, indent=4))


def main():
    global model, half, device, imgsz

    # download weights
    local_path = os.path.join(my_app.data_dir, sly.fs.get_file_name_with_ext(REMOTE_PATH))
    my_app.public_api.file.download(TEAM_ID, REMOTE_PATH, local_path)

    # load model on device
    model, half, device, imgsz = load_model(local_path, imgsz=640, device=DEVICE_STR)
    construct_model_meta(model)

    debug_inference()

    x = 10
    x += 1


#@TODO: alex - test bbox coordinates
#@TODO: augment argument before deploy
#@TODO: log input arguments
#@TODO: save image size to model
#@TODO: save colors to model
#@TODO: fix serve template - debug_inference
#@TODO: or another app serve_cpu?
#@TODO: deploy on custom device: cpu/gpu
if __name__ == "__main__":
    sly.main_wrapper("main", main)