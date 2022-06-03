import json
import os

import supervisely as sly

import supervisely_app.serve.src.nn_utils as nn_utils
import supervisely_app.serve.src.sly_functions as f
import supervisely_app.serve.src.sly_globals as g


@g.my_app.callback("get_output_classes_and_tags")
@sly.timeit
def get_output_classes_and_tags(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=g.meta.to_json())


@g.my_app.callback("get_session_info")
@sly.timeit
def get_session_info(api: sly.Api, task_id, context, state, app_logger):
    info = {
        "app": "YOLOv5 serve",
        "weights": g.final_weights,
        "device": str(g.device),
        "half": str(g.half),
        "input_size": g.imgsz,
        "session_id": task_id,
        "classes_count": len(g.meta.obj_classes),
        "tags_count": len(g.meta.tag_metas),
        "sliding_window_support": True,
        "apply_nn_videos_support": True
    }
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=info)


@g.my_app.callback("get_custom_inference_settings")
@sly.timeit
def get_custom_inference_settings(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data={"settings": g.default_settings_str})


@g.my_app.callback("inference_image_url")
@sly.timeit
def inference_image_url(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})

    image_url = state["image_url"]
    ext = sly.fs.get_file_ext(image_url)
    if ext == "":
        ext = ".jpg"
    local_image_path = os.path.join(g.my_app.data_dir, sly.rand_str(15) + ext)

    sly.fs.download(image_url, local_image_path)
    ann_json = f.inference_image_path(image_path=local_image_path, project_meta=g.meta,
                                      context=context, state=state, app_logger=app_logger)
    sly.fs.silent_remove(local_image_path)

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=ann_json)


@g.my_app.callback("inference_image_id")
@sly.timeit
def inference_image_id(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_id = state["image_id"]
    image_info = api.image.get_info_by_id(image_id)
    image_path = os.path.join(g.my_app.data_dir, sly.rand_str(10) + image_info.name)
    api.image.download_path(image_id, image_path)
    ann_json = f.inference_image_path(image_path=image_path, project_meta=g.meta,
                                      context=context, state=state, app_logger=app_logger)
    sly.fs.silent_remove(image_path)
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=ann_json)


@g.my_app.callback("inference_batch_ids")
@sly.timeit
def inference_batch_ids(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    ids = state["batch_ids"]
    infos = api.image.get_info_by_id_batch(ids)
    paths = []
    for info in infos:
        paths.append(os.path.join(g.my_app.data_dir, sly.rand_str(10) + info.name))
    api.image.download_paths(infos[0].dataset_id, ids, paths)

    results = []
    for image_path in paths:
        ann_json = f.inference_image_path(image_path=image_path, project_meta=g.meta,
                                          context=context, state=state, app_logger=app_logger)
        results.append(ann_json)
        sly.fs.silent_remove(image_path)

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=results)


def debug_inference():
    image = sly.image.read("./data/images/bus.jpg")  # RGB
    ann = nn_utils.inference(g.model, g.half, g.device, g.imgsz, stride=g.stride, image=image, meta=g.meta,
                             debug_visualization=True)
    print(json.dumps(ann, indent=4))


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.TEAM_ID,
        "context.workspaceId": g.WORKSPACE_ID,
        "modal.state.modelWeightsOptions": g.modelWeightsOptions,
        "modal.state.modelSize": g.pretrained_weights,
        "modal.state.weightsPath": g.custom_weights
    })

    f.preprocess()
    sly.logger.info("🟩 Model has been successfully deployed")

    g.my_app.run()


# @TODO: move inference methods to SDK
# @TODO: augment inference
# @TODO: https://pypi.org/project/cachetools/

if __name__ == "__main__":
    sly.main_wrapper("main", main)
