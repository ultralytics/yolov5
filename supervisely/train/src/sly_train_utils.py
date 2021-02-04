import os
import sys
import yaml
from pathlib import Path
import supervisely_lib as sly

import sly_train_globals as g
from sly_train_globals import my_app, api, task_id, team_id


def init_script_arguments(state, yolov5_format_dir, input_project_name):
    global local_artifacts_dir, remote_artifacts_dir
    data_path = os.path.join(yolov5_format_dir, 'data_config.yaml')
    sys.argv.extend(["--data", data_path])

    try:
        hyp_content = yaml.safe_load(state["hyp"][state["hypRadio"]])
        hyp = os.path.join(my_app.data_dir, 'hyp.custom.yaml')
        with open(hyp, 'w') as f:
            f.write(state["hyp"][state["hypRadio"]])
    except yaml.YAMLError as e:
        sly.logger.error(repr(e))
        api.app.set_field(task_id, "state.started", False)
        return
    sys.argv.extend(["--hyp", hyp])

    weights = ""  # random (not tested)
    if state["modelWeightsOptions"] == 1:
        weights = state["pretrainedWeights"]
        cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../models', f"{state['modelSize']}.yaml")
        sys.argv.extend(["--cfg", cfg])
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

    if "workers" in state:
        sys.argv.extend(["--workers", str(state["workers"])])

    training_dir = os.path.join(my_app.data_dir, 'experiment', input_project_name)
    experiment_name = str(task_id)
    local_artifacts_dir = os.path.join(training_dir, experiment_name)
    _exp_index = 1
    while sly.fs.dir_exists(local_artifacts_dir):
        experiment_name = "{}_{:03d}".format(task_id, _exp_index)
        local_artifacts_dir = os.path.join(training_dir, experiment_name)
        _exp_index += 1
    g.local_artifacts_dir = local_artifacts_dir

    sys.argv.extend(["--project", training_dir])
    sys.argv.extend(["--name", experiment_name])

    sys.argv.append("--sly")

    remote_experiment_name = str(task_id)
    remote_artifacts_dir = os.path.join("/yolov5_train", input_project_name, remote_experiment_name)
    _exp_index = 1
    while api.file.dir_exists(team_id, remote_artifacts_dir):
        remote_experiment_name = "{}_{:03d}".format(task_id, _exp_index)
        remote_artifacts_dir = os.path.join("/yolov5_train", input_project_name, remote_experiment_name)
        _exp_index += 1
    g.remote_artifacts_dir = remote_artifacts_dir


def send_epoch_log(epoch, epochs):
    fields = [
        {"field": "data.progressName", "payload": "Epoch"},
        {"field": "data.currentProgressLabel", "payload": epoch},
        {"field": "data.totalProgressLabel", "payload": epochs},
        {"field": "data.currentProgress", "payload": epoch},
        {"field": "data.totalProgress", "payload": epochs},
    ]
    api.app.set_fields(task_id, fields)


def upload_label_vis():
    paths = [x for x in Path(local_artifacts_dir).glob('labels*.jpg') if x.exists()]
    _upload_data_vis(f"data.labelsVis", paths, len(paths))


def upload_pred_vis():
    paths = [x for x in Path(local_artifacts_dir).glob('test*.jpg') if x.exists()]
    _paths = [str(path) for path in paths]
    _paths.sort()
    sync_bindings = []
    for batch in sly.batched(_paths, 2):
        sync_bindings.append([sly.fs.get_file_name_with_ext(batch[0]), sly.fs.get_file_name_with_ext(batch[1])])
    api.task.set_field(task_id, "data.syncBindings", sync_bindings)
    _upload_data_vis("data.predVis", paths, 2)


def upload_train_data_vis():
    paths = [x for x in Path(local_artifacts_dir).glob('train*.jpg') if x.exists()]
    cnt_columns = len(paths)
    if cnt_columns > 3 and cnt_columns <= 9:
        cnt_columns = 3
    elif cnt_columns > 9:
        cnt_columns = 5
    _upload_data_vis("data.vis", paths, cnt_columns)


def _upload_data_vis(field, paths, cnt_columns):
    annotations = {}
    grid_layout = [[] for i in range(cnt_columns)]
    _paths = [str(path) for path in paths]
    _paths.sort()
    for idx, file_path in enumerate(_paths):
        remote_file_path = os.path.join(remote_artifacts_dir, file_path.replace(local_artifacts_dir, '').lstrip("/"))
        file_info = api.file.upload(team_id, file_path, remote_file_path)
        annotations[file_info.name] = {
            "url": file_info.full_storage_url,
            "name": file_info.name,
            "figures": []
        }
        grid_layout[idx % cnt_columns].append(file_info.name)

    fields = [
        {"field": f"{field}.content.annotations", "payload": annotations},
        {"field": f"{field}.content.layout", "payload": grid_layout},
    ]
    api.app.set_fields(task_id, fields)
