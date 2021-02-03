import os
import sys
import supervisely_lib as sly
from supervisely_lib._utils import sizeof_fmt

from sly_train_val_split import train_val_split
from sly_init_ui import init_input_project, init_classes_stats, init_random_split, init_model_settings, \
    init_training_hyperparameters, _load_file as load_hyp
from sly_prepare_data import filter_and_transform_labels
from sly_train_utils import init_script_arguments, empty_gallery
import sly_train_utils
from sly_metrics_utils import init_metrics
import sly_metrics_utils

my_app = sly.AppService()

TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
PROJECT_ID = int(os.environ['modal.state.slyProjectId'])

PROJECT = None
META = None

CNT_GRID_COLUMNS = 3


@my_app.callback("restore_hyp")
@sly.timeit
def restore_hyp(api: sly.Api, task_id, context, state, app_logger):
    api.task.set_field(task_id, "state.hyp", {
        "scratch": load_hyp('data/hyp.scratch.yaml'),
        "finetune": load_hyp('data/hyp.finetune.yaml'),
    })


@my_app.callback("train")
@sly.timeit
def train(api: sly.Api, task_id, context, state, app_logger):
    api.app.set_field(task_id, "state.activeNames", ["logs", "labels", "train", "pred", "metrics"])

    sly_train_utils.task_id = task_id
    sly_train_utils.api = api
    sly_train_utils.TEAM_ID = TEAM_ID
    sly_metrics_utils.task_id = task_id
    sly_metrics_utils.api = api

    project_dir = os.path.join(my_app.data_dir, "sly_project")
    sly.fs.mkdir(project_dir)
    sly.fs.clean_dir(project_dir)  # useful for debug

    progress = sly.Progress("Download data (using cache)", PROJECT.items_count * 2, ext_logger=app_logger)
    def progress_cb(count):
        progress.iters_done_report(count)
        fields = [
            {"field": "data.progressName", "payload": progress.message},
            {"field": "data.currentProgressLabel", "payload": progress.current},
            {"field": "data.totalProgressLabel", "payload": progress.total},
            {"field": "data.currentProgress", "payload": progress.current},
            {"field": "data.totalProgress", "payload": progress.total},
        ]
        api.app.set_fields(task_id, fields)

    progress_cb(0)
    sly.download_project_optimized(api, project_dir, PROJECT_ID, cache=my_app.cache, progress_cb=progress_cb)

    train_split, val_split = train_val_split(project_dir, state)
    train_classes = state["selectedClasses"]
    yolov5_format_dir = os.path.join(my_app.data_dir, "train_data")
    sly.fs.mkdir(yolov5_format_dir)
    sly.fs.clean_dir(yolov5_format_dir)  # useful for debug

    progress = sly.Progress("Convert Supervisely to YOLOv5 format", len(train_split) + len(val_split),
                            ext_logger=app_logger)
    progress_cb(0)
    filter_and_transform_labels(project_dir, META, train_classes, train_split, val_split, yolov5_format_dir,
                                progress_cb)

    local_artifacts_dir, remote_artifacts_dir = \
        init_script_arguments(state, yolov5_format_dir, my_app.data_dir, PROJECT.name, task_id)

    # download initial weights from team files
    if state["modelWeightsOptions"] == 2:  # transfer learning from custom weights
        weights_path_remote = state["weightsPath"]
        weights_path_local = os.path.join(my_app.data_dir, sly.fs.get_file_name_with_ext(weights_path_remote))
        file_info = api.file.get_info_by_path(TEAM_ID, weights_path_remote)
        cache_path = my_app.cache.get_storage_path(file_info.hash)
        if cache_path is None:
            api.file.download(TEAM_ID, weights_path_remote, weights_path_local)  # TODO: progress bar
            my_app.cache.write_object(weights_path_local, file_info.hash)

    progress = sly.Progress("YOLOv5: Scanning data ", 1, ext_logger=app_logger)
    def progress_cb(count):
        progress.iters_done_report(count)
        fields = [
            {"field": "data.progressName", "payload": progress.message},
            {"field": "data.currentProgressLabel", "payload": progress.current},
            {"field": "data.totalProgressLabel", "payload": progress.total},
            {"field": "data.currentProgress", "payload": progress.current},
            {"field": "data.totalProgress", "payload": progress.total},
        ]
        api.app.set_fields(task_id, fields)

    progress_cb(1)

    import train
    train.main()

    progress = sly.Progress("Download data (using cache)", PROJECT.items_count * 2, ext_logger=app_logger)

    upload_progress = [None]

    def _print_progress(monitor, upload_progress):
        if len(upload_progress) == 0:
            upload_progress.append(sly.Progress(message="Upload {!r}".format(local_path),
                                                total_cnt=monitor.len,
                                                ext_logger=app_logger,
                                                is_size=True))
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
        if api.file.exists(TEAM_ID, remote_path):
            continue
        upload_progress.pop(0)
        api.file.upload(TEAM_ID, local_path, remote_path, lambda m: _print_progress(m, upload_progress))

    progress = sly.Progress("Finished, app is stopped automatically", 1, ext_logger=app_logger)
    progress_cb(1)

    file_info = api.file.get_info_by_path(TEAM_ID, os.path.join(remote_artifacts_dir, 'results.png'))
    fields = [
        {"field": "data.outputUrl", "payload": api.file.get_url(file_info.id)},
        {"field": "data.outputName", "payload": remote_artifacts_dir},
    ]
    api.app.set_fields(task_id, fields)

    my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": TEAM_ID,
        "context.workspaceId": WORKSPACE_ID,
        "modal.state.slyProjectId": PROJECT_ID,
    })

    data = {}
    state = {}

    data["taskId"] = my_app.task_id
    global PROJECT, META
    PROJECT, META = init_input_project(my_app.public_api, PROJECT_ID, data)
    init_classes_stats(my_app.public_api, PROJECT_ID, META, data, state)

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
    my_app.run(template_path, data, state)


from functools import partial
def func(a, b, progress):
    print("a = ", a)
    print("b = ", b)
    progress.iters_done_report(1)


def update_progress(count, api, task_id, progress):
    progress.iters_done_report(count)
    if progress.need_report():  # @TODO: decrease number of updates
        fields = [
            {"field": "data.progressName", "payload": progress.message},
            {"field": "data.currentProgressLabel", "payload": progress.current_label},
            {"field": "data.totalProgressLabel", "payload": progress.total_label},
            {"field": "data.currentProgress", "payload": progress.current},
            {"field": "data.totalProgress", "payload": progress.total},
        ]
        api.app.set_fields(task_id, fields)


def debug_download_progress():
    team_id = 7
    remote_path = "/yolov5_train/coco128_002/2390/weights/best.pt"
    sly.fs.ensure_base_path(remote_path)
    sly.fs.silent_remove(remote_path)
    #my_app.public_api.file.download(team_id, remote_path, remote_path, my_app.cache, progress_cb)

    file_info = my_app.public_api.file.get_info_by_path(team_id, remote_path)
    progress = sly.Progress("doing", file_info.sizeb, is_size=True)
    progress_cb = partial(update_progress, api=my_app.public_api, task_id=my_app.task_id, progress=progress)
    my_app.public_api.file.download(team_id, remote_path, remote_path, None, progress_cb)
    print(sly.fs.file_exists(remote_path))
    pass


# @TODO: как поставить ограничение на номер версии инстанса?
# @TODO: download project optimized - cache_path + rename - fix bug
# @TODO: download progress bar for weights in SDK
# @TODO: add files for remote debug (docker-compose.yaml)
# @TODO: train == val - handle case in data_config.yaml to avoid data duplication
# @TODO: Double progress: progress bar iterations, progress bar upload
if __name__ == "__main__":
    debug_download_progress()

    #sly.main_wrapper("main", main)
