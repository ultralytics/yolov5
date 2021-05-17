import os
import sys
import yaml
import time
from pathlib import Path
import supervisely_lib as sly

import sly_train_globals as g
from sly_train_globals import my_app, api, task_id, team_id


def init_script_arguments(state, yolov5_format_dir, input_project_name):
    global local_artifacts_dir, remote_artifacts_dir
    sys.argv.append("--sly")

    data_path = os.path.join(yolov5_format_dir, 'data_config.yaml')
    sys.argv.extend(["--data", data_path])

    hyp_content = yaml.safe_load(state["hyp"][state["hypRadio"]])
    hyp = os.path.join(my_app.data_dir, 'hyp.custom.yaml')
    with open(hyp, 'w') as f:
        f.write(state["hyp"][state["hypRadio"]])
    sys.argv.extend(["--hyp", hyp])

    if state["weightsInitialization"] == "coco":
        model_name = state['selectedModel'].lower()
        _sub_path = "models/hub" if model_name.endswith('6') else "models"
        cfg = os.path.join(g.root_source_dir, _sub_path, f"{model_name}.yaml")
        sys.argv.extend(["--cfg", cfg])
    sys.argv.extend(["--weights", state["weightsPath"]])

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
    if state["optimizer"] == "Adam":
        sys.argv.append("--adam")

    sys.argv.extend(["--metrics_period", str(state["metricsPeriod"])])
    sys.argv.extend(["--project", g.runs_dir])
    sys.argv.extend(["--name", g.experiment_name])


def send_epoch_log(epoch, epochs, progress):
    progress.set_current_value(epoch)
    fields = [
        {"field": "data.progressName", "payload": "Epoch"},
        {"field": "data.currentProgressLabel", "payload": epoch},
        {"field": "data.totalProgressLabel", "payload": epochs},
        {"field": "data.currentProgress", "payload": epoch},
        {"field": "data.totalProgress", "payload": epochs},
    ]
    api.app.set_fields(task_id, fields)


def upload_label_vis():
    paths = [x for x in Path(g.local_artifacts_dir).glob('labels*.jpg') if x.exists()]
    _upload_data_vis(f"data.labelsVis", paths, len(paths))


def upload_pred_vis():
    submitted = False
    for i in range(5):
        paths = [x for x in Path(g.local_artifacts_dir).glob('test*.jpg') if x.exists()]
        if len(paths) % 2 != 0:
            time.sleep(3)  # wait while thread in YOLOv5 script produce visualization: test batch + prediction
            continue
        _paths = [str(path) for path in paths]
        _paths.sort()
        sync_bindings = []
        for batch in sly.batched(_paths, 2):
            sync_bindings.append([sly.fs.get_file_name_with_ext(batch[0]), sly.fs.get_file_name_with_ext(batch[1])])
        api.task.set_field(task_id, "data.syncBindings", sync_bindings)
        _upload_data_vis("data.predVis", paths, 2)
        submitted = True
        break

    if submitted is False:
        sly.logger.warn("Test batch visualizations (labels + predictions) are not ready, see them in artifacts "
                        "directory after training ")
        pass


def upload_train_data_vis():
    paths = [x for x in Path(g.local_artifacts_dir).glob('train*.jpg') if x.exists()]
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
        remote_file_path = os.path.join(g.remote_artifacts_dir, file_path.replace(g.local_artifacts_dir, '').lstrip("/"))
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
