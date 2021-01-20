import os
import sys
import supervisely_lib as sly
from supervisely_lib._utils import sizeof_fmt

from sly_train_val_split import train_val_split
from sly_init_ui import init_input_project, init_classes_stats, init_random_split, init_model_settings, \
     init_training_hyperparameters
from sly_prepare_data import filter_and_transform_labels
from sly_train_utils import init_script_arguments
import sly_train_utils

my_app = sly.AppService()


TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
PROJECT_ID = int(os.environ['modal.state.slyProjectId'])

PROJECT = None
META = None

CNT_GRID_COLUMNS = 3
empty_gallery = {
    "content": {
        "projectMeta": sly.ProjectMeta().to_json(),
        "annotations": {},
        "layout": []
    }
}


@my_app.callback("train")
@sly.timeit
def train(api: sly.Api, task_id, context, state, app_logger):
    sly_train_utils.task_id = task_id
    sly_train_utils.api = api
    sly_train_utils.TEAM_ID = TEAM_ID

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

    progress = sly.Progress("Convert Supervisely to YOLOv5 format", len(train_split) + len(val_split), ext_logger=app_logger)
    progress_cb(0)
    filter_and_transform_labels(project_dir, META, train_classes, train_split, val_split, yolov5_format_dir, progress_cb)

    local_artifacts_dir, remote_artifacts_dir = \
        init_script_arguments(state, yolov5_format_dir, my_app.data_dir, PROJECT.name, task_id)

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
        if sly.image.has_valid_ext(local_path):
            continue
        upload_progress.pop(0)
        remote_path = os.path.join(remote_artifacts_dir, local_path.replace(local_artifacts_dir, '').lstrip("/"))
        api.file.upload(TEAM_ID, local_path, remote_path, lambda m: _print_progress(m, upload_progress))

    progress = sly.Progress("Finished, app will be stopped automatically", 1, ext_logger=app_logger)
    progress_cb(1)

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
    state["epochs"] = 2  # @TODO: uncomment for debug
    state["activeNames"] = ["logs", "labels", "train"]
    data["vis"] = empty_gallery
    data["labelsVis"] = empty_gallery

    data["progressName"] = ""
    data["currentProgress"] = 0
    data["totalProgress"] = 0

    template_path = os.path.join(os.path.dirname(sys.argv[0]), 'supervisely/train/src/gui.html')
    my_app.run(template_path, data, state)


#@TODO: train == val - handle case in data_config.yaml to avoid data duplication
#@TODO: --hyp file - (scratch or finetune ...) - all params to advanced settings in UI
#@TODO: disable all widget when start :disabled="state.started === True"
#@TODO: save direct link to session in directory
if __name__ == "__main__":
    sly.main_wrapper("main", main)