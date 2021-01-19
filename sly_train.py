import os
import sys
import supervisely_lib as sly

from sly_train_val_split import train_val_split
from sly_init_ui import init_input_project, init_classes_stats, init_random_split, init_model_settings, \
     init_training_hyperparameters
from sly_prepare_data import filter_and_transform_labels
from sly_train_utils import init_script_arguments

my_app = sly.AppService()


TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
PROJECT_ID = int(os.environ['modal.state.slyProjectId'])

PROJECT = None
META = None


@my_app.callback("train")
@sly.timeit
def train(api: sly.Api, task_id, context, state, app_logger):
    project_dir = os.path.join(my_app.data_dir, "sly_project")
    sly.fs.mkdir(project_dir)
    sly.fs.clean_dir(project_dir)  # useful for debug
    sly.download_project_optimized(api, project_dir, PROJECT_ID, cache=my_app.cache, logger=app_logger)

    train_split, val_split = train_val_split(project_dir, state)
    train_classes = state["selectedClasses"]
    yolov5_format_dir = os.path.join(my_app.data_dir, "train_data")
    sly.fs.mkdir(yolov5_format_dir)
    sly.fs.clean_dir(yolov5_format_dir)  # useful for debug
    filter_and_transform_labels(project_dir, META, train_classes, train_split, val_split, yolov5_format_dir)

    local_artifacts_dir, remote_artifacts_dir = \
        init_script_arguments(state, yolov5_format_dir, my_app.data_dir, PROJECT.name, task_id)

    import train
    train.main()

    api.file.upload_directory(TEAM_ID, local_artifacts_dir, remote_artifacts_dir)


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
    state["epochs"] = 1  # @TODO: uncomment for debug

    template_path = os.path.join(os.path.dirname(sys.argv[0]), 'supervisely/train/src/gui.html')
    my_app.run(template_path, data, state)


#@TODO: train == val - handle case in data_config.yaml to avoid data duplication
#@TODO: --hyp file - (scratch or finetune ...) - all params to advanced settings in UI
#@TODO: disable all widget when start :disabled="state.started === True"
if __name__ == "__main__":
    sly.main_wrapper("main", main)