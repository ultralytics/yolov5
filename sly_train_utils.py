import os
import sys
import supervisely_lib as sly


def init_script_arguments(state, yolov5_format_dir, app_data_dir, input_project_name, task_id):
    data_path = os.path.join(yolov5_format_dir, 'data_config.yaml')
    sys.argv.extend(["--data", data_path])

    cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', f"{state['modelSize']}.yaml")
    sys.argv.extend(["--cfg", cfg])

    hyp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/hyp.scratch.yaml')
    sys.argv.extend(["--hyp", hyp])

    weights = ""  # random
    if state["modelWeightsOptions"] == 1:
        weights = state["pretrainedWeights"]
    elif state["modelWeightsOptions"] == 2:
        weights = state["weightsPath"]
    sys.argv.extend(["--weights", weights])

    sys.argv.extend(["--epochs", str(state["epochs"])])
    sys.argv.extend(["--batch-size", str(state["batchSize"])])
    sys.argv.extend(["--img-size", str(state["imgSize"]), str(state["imgSize"])])
    if state["multiScale"]:
        sys.argv.append("--multi-scale")
    if state["singleClass"]:
        sys.argv.append("--single-cls")
    sys.argv.extend(["--device", state["device"]])

    training_dir = os.path.join(app_data_dir, 'experiment', input_project_name)
    experiment_name = str(task_id)
    local_artifacts_dir = os.path.join(training_dir, experiment_name)
    _exp_index = 1
    while sly.fs.dir_exists(local_artifacts_dir):
        experiment_name = "{}_{:03d}".format(task_id, _exp_index)
        local_artifacts_dir = os.path.join(training_dir, experiment_name)
        _exp_index += 1

    sys.argv.extend(["--project", training_dir])
    sys.argv.extend(["--name", experiment_name])

    remote_artifacts_dir = os.path.join("/yolov5_train", input_project_name, experiment_name)
    return local_artifacts_dir, remote_artifacts_dir