import os
import sys
import supervisely_lib as sly


#sys.argv.append('--weights')
#sys.argv.append("maxim")
#import train
#train.main()

my_app = sly.AppService()


TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
PROJECT_ID = int(os.environ['modal.state.slyProjectId'])

PROJECT = None
META = None

#@my_app.callback("yolov5_sly_converter")
#@sly.timeit
# def yolov5_sly_converter(api: sly.Api, task_id, context, state, app_logger):
#     pass


def init_input_project(api: sly.Api, data):
    global PROJECT, META
    PROJECT = api.project.get_info_by_id(PROJECT_ID)
    META = sly.ProjectMeta.from_json(api.project.get_meta(PROJECT_ID))
    data["projectId"] = PROJECT_ID
    data["projectName"] = PROJECT.name
    data["projectPreviewUrl"] = api.image.preview_url(PROJECT.reference_image_url, 100, 100)


def init_classes_stats(api: sly.Api, data, state):
    stats = api.project.get_stats(PROJECT_ID)
    class_images = {}
    for item in stats["images"]["objectClasses"]:
        class_images[item["objectClass"]["name"]] = item["total"]
    class_objects = {}
    for item in stats["objects"]["items"]:
        class_objects[item["objectClass"]["name"]] = item["total"]

    classes_json = META.obj_classes.to_json()
    for obj_class in classes_json:
        obj_class["imagesCount"] = class_images[obj_class["title"]]
        obj_class["objectsCount"] = class_objects[obj_class["title"]]

    data["classes"] = classes_json
    state["selectedClasses"] = []

    state["classes"] = len(classes_json) * [True]


def init_random_split(data, state):
    data["randomSplit"] = [
        {"name": "train", "type": "success"},
        {"name": "val", "type": "primary"},
        {"name": "total", "type": "gray"},
    ]
    data["totalImagesCount"] = PROJECT.items_count

    train_percent = 80
    train_count = int(PROJECT.items_count / 100 * train_percent)
    state["randomSplit"] = {
        "count": {
            "total": PROJECT.items_count,
            "train": train_count,
            "val": PROJECT.items_count - train_count
        },
        "percent": {
            "total": 100,
            "train": train_percent,
            "val": 100 - train_percent
        },
        "shareImagesBetweenSplits": False,
        "sliderDisabled": False,
    }


def init_tag_split(state):
    state["trainTagName"] = ""
    state["valTagName"] = ""


def init_model_settings(data, state):
    data["modelSizes"] = [
        {"label": "yolov5s", "config": "yolov5s.yaml", "params": "7.3M"},
        {"label": "yolov5m", "config": "yolov5m.yaml", "params": "21.4M"},
        {"label": "yolov5l", "config": "yolov5l.yaml", "params": "47.0M"},
        {"label": "yolov5x", "config": "yolov5x.yaml", "params": "87.7M"},
    ]
    state["modelSize"] = data["modelSizes"][0]["label"]
    state["modelWeightsOptions"] = 1
    state["pretrainedWeights"] = f'{data["modelSizes"][0]["label"]}.pt'
    state["weightsPath"] = ""


def init_training_hyperparameters(state):
    state["epochs"] = 300
    state["batchSize"] = 16
    state["imgSize"] = 640
    state["multiScale"] = True
    state["singleClass"] = False
    state["device"] = '0'


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": TEAM_ID,
        "context.workspaceId": WORKSPACE_ID,
        "modal.state.slyProjectId": PROJECT_ID,
    })

    data = {}
    state = {}

    data["taskId"] = my_app.task_id
    init_input_project(my_app.public_api, data)
    init_classes_stats(my_app.public_api, data, state)
    state["splitMethod"] = 1
    init_random_split(data, state)
    init_tag_split(state)
    init_model_settings(data, state)
    init_training_hyperparameters(state)

    #my_app.run(initial_events=[{"command": "yolov5_sly_converter"}])
    template_path = os.path.join(os.path.dirname(sys.argv[0]), 'supervisely/train/src/gui.html')
    my_app.run(template_path, data, state)


if __name__ == "__main__":
    sly.main_wrapper("main", main)