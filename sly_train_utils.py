import os
import sys
import yaml
from pathlib import Path
import supervisely_lib as sly

import sly_train_globals as globals


empty_gallery = {
    "content": {
        "projectMeta": sly.ProjectMeta().to_json(),
        "annotations": {},
        "layout": []
    }
}


def init_script_arguments(state, yolov5_format_dir, input_project_name):
    global local_artifacts_dir, remote_artifacts_dir
    data_path = os.path.join(yolov5_format_dir, 'data_config.yaml')
    sys.argv.extend(["--data", data_path])

    cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', f"{state['modelSize']}.yaml")
    sys.argv.extend(["--cfg", cfg])

    try:
        hyp_content = yaml.safe_load(state["hyp"][state["hypRadio"]])
        hyp = os.path.join(globals.app.data_dir, 'hyp.custom.yaml')
        with open(hyp, 'w') as f:
            f.write(state["hyp"][state["hypRadio"]])
    except yaml.YAMLError as e:
        sly.logger.error(repr(e))
        globals.api.app.set_field(globals.task_id, "state.started", False)
        return
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

    if "workers" in state:
        sys.argv.extend(["--workers", str(state["workers"])])

    training_dir = os.path.join(globals.app.data_dir, 'experiment', input_project_name)
    experiment_name = str(globals.task_id)
    globals.local_artifacts_dir = os.path.join(training_dir, experiment_name)
    _exp_index = 1
    while sly.fs.dir_exists(globals.local_artifacts_dir):
        experiment_name = "{}_{:03d}".format(globals.task_id, _exp_index)
        globals.local_artifacts_dir = os.path.join(training_dir, experiment_name)
        _exp_index += 1

    sys.argv.extend(["--project", training_dir])
    sys.argv.extend(["--name", experiment_name])

    sys.argv.append("--sly")

    remote_experiment_name = str(globals.task_id)
    globals.remote_artifacts_dir = os.path.join("/yolov5_train", input_project_name, remote_experiment_name)
    _exp_index = 1
    while globals.api.file.dir_exists(globals.TEAM_ID, globals.remote_artifacts_dir):
        remote_experiment_name = "{}_{:03d}".format(globals.task_id, _exp_index)
        globals.remote_artifacts_dir = os.path.join("/yolov5_train", input_project_name, remote_experiment_name)
        _exp_index += 1


def send_epoch_log(epoch, epochs):
    fields = [
        {"field": "data.progressName", "payload": "Epoch"},
        {"field": "data.currentProgressLabel", "payload": epoch},
        {"field": "data.totalProgressLabel", "payload": epochs},
        {"field": "data.currentProgress", "payload": epoch},
        {"field": "data.totalProgress", "payload": epochs},
    ]
    globals.api.app.set_fields(globals.task_id, fields)


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
    globals.api.task.set_field(globals.task_id, "data.syncBindings", sync_bindings)
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
        file_info = globals.api.file.upload(globals.TEAM_ID, file_path, remote_file_path)
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
    globals.api.app.set_fields(globals.task_id, fields)
