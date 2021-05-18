import os
import supervisely_lib as sly

import sly_train_globals as g

from sly_train_globals import \
    my_app, task_id, \
    team_id, workspace_id, project_id, \
    root_source_dir, scratch_str, finetune_str

import ui as ui
from sly_train_utils import init_script_arguments
from sly_utils import get_progress_cb, upload_artifacts
from splits import get_train_val_sets, verify_train_val_sets
import yolov5_format as yolov5_format
from architectures import prepare_weights
from artifacts import set_task_output
import train as train_yolov5


@my_app.callback("restore_hyp")
@sly.timeit
def restore_hyp(api: sly.Api, task_id, context, state, app_logger):
    api.task.set_field(task_id, "state.hyp", {
        "scratch": scratch_str,
        "finetune": finetune_str,
    })


@my_app.callback("train")
@sly.timeit
def train(api: sly.Api, task_id, context, state, app_logger):
    try:
        prepare_weights(state)

        # prepare directory for original Supervisely project
        project_dir = os.path.join(my_app.data_dir, "sly_project")
        sly.fs.mkdir(project_dir, remove_content_if_exists=True)  # clean content for debug, has no effect in prod

        # download and preprocess Sypervisely project (using cache)
        download_progress = get_progress_cb("Download data (using cache)", g.project_info.items_count * 2)
        sly.download_project(api, project_id, project_dir, cache=my_app.cache, progress_cb=download_progress)

        # preprocessing: transform labels to bboxes, filter classes, ...
        sly.Project.to_detection_task(project_dir, inplace=True)
        train_classes = state["selectedClasses"]
        sly.Project.remove_classes_except(project_dir, classes_to_keep=train_classes, inplace=True)
        if state["unlabeledImages"] == "ignore":
            sly.Project.remove_items_without_objects(project_dir, inplace=True)

        # split to train / validation sets (paths to images and annotations)
        train_set, val_set = get_train_val_sets(project_dir, state)
        verify_train_val_sets(train_set, val_set)
        sly.logger.info(f"Train set: {len(train_set)} images")
        sly.logger.info(f"Val set: {len(val_set)} images")

        # prepare directory for data in YOLOv5 format (nn will use it for training)
        train_data_dir = os.path.join(my_app.data_dir, "train_data")
        sly.fs.mkdir(train_data_dir, remove_content_if_exists=True)  # clean content for debug, has no effect in prod

        # convert Supervisely project to YOLOv5 format
        progress_cb = get_progress_cb("Convert Supervisely to YOLOv5 format", len(train_set) + len(val_set))
        yolov5_format.transform(project_dir, train_data_dir, train_set, val_set, progress_cb)

        # init sys.argv for main training script
        init_script_arguments(state, train_data_dir, g.project_info.name)

        # start train script
        api.app.set_field(task_id, "state.activeNames", ["labels", "train", "pred", "metrics"])  # "logs",
        get_progress_cb("YOLOv5: Scanning data ", 1)(1)
        train_yolov5.main()

        # upload artifacts directory to Team Files
        upload_artifacts(g.local_artifacts_dir, g.remote_artifacts_dir)
        set_task_output()
    except Exception as e:
        my_app.show_modal_window(f"Oops! Something went wrong, please try again or contact tech support. "
                                 f"Find more info in the app logs. Error: {repr(e)}", level="error")
        api.app.set_field(task_id, "state.started", False)

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

    my_app.compile_template(g.root_source_dir)

    # init data for UI widgets
    ui.init(data, state)

    my_app.run(data=data, state=state)


# New features:
# @TODO: resume training
# @TODO: save checkpoint every N-th epochs
if __name__ == "__main__":
    sly.main_wrapper("main", main)
