import os
import sys
import supervisely_lib as sly

import sly_train_globals as globals
from supervisely_lib._utils import sizeof_fmt


from sly_train_val_split import train_val_split
from sly_init_ui import init_input_project, init_classes_stats, init_random_split, init_model_settings, \
    init_training_hyperparameters, _load_file as load_hyp
from sly_prepare_data import filter_and_transform_labels
from sly_train_utils import init_script_arguments, empty_gallery
from sly_metrics_utils import init_metrics
from sly_utils import update_progress, get_progress_cb


PROJECT = None
META = None
CNT_GRID_COLUMNS = 3


@globals.app.callback("restore_hyp")
@sly.timeit
def restore_hyp(api: sly.Api, task_id, context, state, app_logger):
    api.task.set_field(task_id, "state.hyp", {
        "scratch": load_hyp('data/hyp.scratch.yaml'),
        "finetune": load_hyp('data/hyp.finetune.yaml'),
    })


@globals.app.callback("train")
@sly.timeit
def train(api: sly.Api, task_id, context, state, app_logger):
    api.app.set_field(task_id, "state.activeNames", ["logs", "labels", "train", "pred", "metrics"])

    # prepare directory for original Supervisely project
    project_dir = os.path.join(globals.app.data_dir, "sly_project")
    sly.fs.mkdir(project_dir)
    sly.fs.clean_dir(project_dir)  # useful for debug

    # download Sypervisely project (using cache)
    sly.download_project(api, globals.PROJECT_ID, project_dir, cache=globals.app.cache,
                         progress_cb=get_progress_cb("Download data (using cache)", PROJECT.items_count * 2))

    # prepare directory for transformed data (nn will use it for training)
    yolov5_format_dir = os.path.join(globals.app.data_dir, "train_data")
    sly.fs.mkdir(yolov5_format_dir)
    sly.fs.clean_dir(yolov5_format_dir)  # useful for debug

    # split data to train/val sets, filter objects by classes, convert Supervisely project to YOLOv5 format(COCO)
    train_split, val_split = train_val_split(project_dir, state)
    train_classes = state["selectedClasses"]
    progress_cb = get_progress_cb("Convert Supervisely to YOLOv5 format", PROJECT.items_count)
    filter_and_transform_labels(project_dir, META, train_classes, train_split, val_split, yolov5_format_dir, progress_cb)

    # download initial weights from team files
    if state["modelWeightsOptions"] == 2:  # transfer learning from custom weights
        weights_path_remote = state["weightsPath"]
        weights_path_local = os.path.join(globals.app.data_dir, sly.fs.get_file_name_with_ext(weights_path_remote))
        file_info = globals.api.file.get_info_by_path(globals.TEAM_ID, weights_path_remote)
        globals.api.file.download(globals.TEAM_ID, weights_path_remote, weights_path_local, globals.app.cache,
                                  progress_cb=get_progress_cb("Download weights", file_info.sizeb))

    # init sys.argv for main training script
    init_script_arguments(state, yolov5_format_dir, PROJECT.name)

    progress_cb = get_progress_cb("YOLOv5: Scanning data ", 1)
    progress_cb(1)

    import train
    train.main()

    progress = sly.Progress("Download data (using cache)", PROJECT.items_count * 2)
    upload_progress = [None]

    def _print_progress(monitor, upload_progress):
        if len(upload_progress) == 0:
            upload_progress.append(sly.Progress(message="Upload {!r}".format(local_path), total_cnt=monitor.len, is_size=True))
        upload_progress[0].set_current_value(monitor.bytes_read)
        progress = upload_progress[0]
        if progress.need_report():
            fields = [
                {"field": "data.progressName", "payload": progress.message},
                {"field": "data.currentProgressLabel", "payload": sizeof_fmt(progress.current)},
                {"field": "data.totalProgressLabel", "payload": sizeof_fmt(progress.total)},
                {"field": "data.currentProgress", "payload": progress.current},
                {"field": "data.totalProgress", "payload": progress.total},
            ]
            api.app.set_fields(task_id, fields)

    local_files = sly.fs.list_files_recursively(local_artifacts_dir)
    for local_path in local_files:
        remote_path = os.path.join(remote_artifacts_dir, local_path.replace(local_artifacts_dir, '').lstrip("/"))
        if api.file.exists(globals.TEAM_ID, remote_path):
            continue
        upload_progress.pop(0)
        api.file.upload(globals.TEAM_ID, local_path, remote_path, lambda m: _print_progress(m, upload_progress))

    progress = sly.Progress("Finished, app is stopped automatically", 1)
    progress_cb(1)

    file_info = api.file.get_info_by_path(globals.TEAM_ID, os.path.join(remote_artifacts_dir, 'results.png'))
    fields = [
        {"field": "data.outputUrl", "payload": api.file.get_url(file_info.id)},
        {"field": "data.outputName", "payload": remote_artifacts_dir},
    ]
    api.app.set_fields(task_id, fields)

    globals.my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": globals.TEAM_ID,
        "context.workspaceId": globals.WORKSPACE_ID,
        "modal.state.slyProjectId": globals.PROJECT_ID,
    })

    data = {}
    state = {}

    data["taskId"] = globals.app.task_id
    global PROJECT, META
    PROJECT, META = init_input_project(globals.app.public_api, globals.PROJECT_ID, data)
    init_classes_stats(globals.app.public_api, globals.PROJECT_ID, META, data, state)

    init_random_split(PROJECT, data, state)
    init_model_settings(data, state)
    init_training_hyperparameters(state)

    state["started"] = False
    state["activeNames"] = []

    data["vis"] = empty_gallery
    data["labelsVis"] = empty_gallery #{} #empty_gallery
    data["predVis"] = empty_gallery

    data["progressName"] = ""
    data["currentProgress"] = 0
    data["totalProgress"] = 0
    data["syncBindings"] = []
    data["outputUrl"] = ""
    data["outputName"] = ""

    init_metrics(data)

    template_path = os.path.join(os.path.dirname(sys.argv[0]), 'supervisely/train/src/gui.html')
    globals.app.run(template_path, data, state)


# @TODO: train == val - handle case in data_config.yaml to avoid data duplication
# @TODO: continue training
if __name__ == "__main__":
    sly.main_wrapper("main", main)
