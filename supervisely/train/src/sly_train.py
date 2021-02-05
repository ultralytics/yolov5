import os
import sys
from pathlib import Path
import supervisely_lib as sly

from sly_train_globals import init_project_info_and_meta, \
                              my_app, task_id, \
                              team_id, workspace_id, project_id

# to import correct values
# project_info, project_meta, \
# local_artifacts_dir, remote_artifacts_dir
import sly_train_globals as g

from sly_train_val_split import train_val_split
import sly_init_ui as ui
from sly_prepare_data import filter_and_transform_labels
from sly_train_utils import init_script_arguments
from sly_utils import get_progress_cb, load_file_as_string, upload_artifacts

root_project_path = str(Path(os.path.realpath(__file__)).parents[3])
sly.logger.info(f"Root project directory: {root_project_path}")
sys.path.append(root_project_path)
import train as train_yolov5
import test as test_yolov5
print("Check imports: ", test_yolov5.test_original)


@my_app.callback("restore_hyp")
@sly.timeit
def restore_hyp(api: sly.Api, task_id, context, state, app_logger):
    api.task.set_field(task_id, "state.hyp", {
        "scratch": load_file_as_string("data/hyp.scratch.yaml"),
        "finetune": load_file_as_string("data/hyp.finetune.yaml"),
    })


@my_app.callback("train")
@sly.timeit
def train(api: sly.Api, task_id, context, state, app_logger):
    api.app.set_field(task_id, "state.activeNames", ["logs", "labels", "train", "pred", "metrics"])

    # prepare directory for original Supervisely project
    project_dir = os.path.join(my_app.data_dir, "sly_project")
    sly.fs.mkdir(project_dir)
    sly.fs.clean_dir(project_dir)  # useful for debug, has no effect in production

    # download Sypervisely project (using cache)
    sly.download_project(api, project_id, project_dir, cache=my_app.cache,
                         progress_cb=get_progress_cb("Download data (using cache)", g.project_info.items_count * 2))

    # prepare directory for transformed data (nn will use it for training)
    yolov5_format_dir = os.path.join(my_app.data_dir, "train_data")
    sly.fs.mkdir(yolov5_format_dir)
    sly.fs.clean_dir(yolov5_format_dir)  # useful for debug, has no effect in production

    # split data to train/val sets, filter objects by classes, convert Supervisely project to YOLOv5 format(COCO)
    train_split, val_split = train_val_split(project_dir, state)
    train_classes = state["selectedClasses"]
    progress_cb = get_progress_cb("Convert Supervisely to YOLOv5 format", g.project_info.items_count)
    filter_and_transform_labels(project_dir, train_classes, train_split, val_split, yolov5_format_dir, progress_cb)

    # download initial weights from team files
    if state["modelWeightsOptions"] == 2:  # transfer learning from custom weights
        weights_path_remote = state["weightsPath"]
        weights_path_local = os.path.join(my_app.data_dir, sly.fs.get_file_name_with_ext(weights_path_remote))
        file_info = api.file.get_info_by_path(team_id, weights_path_remote)
        api.file.download(team_id, weights_path_remote, weights_path_local, my_app.cache,
                          progress_cb=get_progress_cb("Download weights", file_info.sizeb, is_size=True))

    # init sys.argv for main training script
    init_script_arguments(state, yolov5_format_dir, g.project_info.name)

    # start train script
    get_progress_cb("YOLOv5: Scanning data ", 1)(1)
    train_yolov5.main()

    # upload artifacts directory to Team Files
    upload_artifacts(g.local_artifacts_dir, g.remote_artifacts_dir)

    # show path to the artifact directory in Team Files
    ui.set_output()

    # stop application
    get_progress_cb("Finished, app is stopped automatically", 1)(1)
    my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": team_id,
        "context.workspaceId": workspace_id,
        "modal.state.slyProjectId": project_id,
    })

    data = {}
    state = {}
    data["taskId"] = task_id

    # read project information and meta (classes + tags)
    init_project_info_and_meta()

    # init data for UI widgets
    ui.init(data, state)

    my_app.run(data=data, state=state)


# @TODO: train == val - handle case in data_config.yaml to avoid data duplication
# @TODO: resume training
# @TODO: fix upload directory (number of progress updates)
# @TODO: repeat dataset (for small lemons)
# @TODO: chart refresh freezes page
if __name__ == "__main__":
    sly.main_wrapper("main", main)
