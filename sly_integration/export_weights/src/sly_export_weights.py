import supervisely as sly
# from supervisely.io.fs import download, file_exists, get_file_name, get_file_name_with_ext
import os
import pathlib
import torch
import torch.nn as nn

from pathlib import Path
import yaml
from sly_globals import my_app, TEAM_ID, WORKSPACE_ID, customWeightsPath, device, image_size, batch_size, grid, args, ts

import sys

root_source_path = str(pathlib.Path(sys.argv[0]).parents[3])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)

app_source_path = str(pathlib.Path(sys.argv[0]).parents[0])
sly.logger.info(f"App root source directory: {app_source_path}")
sys.path.append(app_source_path)

import models
from app_utils import download_weights
from utils.general import check_img_size  # , colorstr, check_requirements, file_size, set_logging
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from models.common import Conv, DWConv


def export_to_torch_script(weights, img, model):
    try:
        f = weights.replace('.pt', '.torchscript.pt')  # filename
        ts = torch.jit.trace(model, img, strict=False)
        ts.save(f)
    except Exception as e:
        print(f'export failure: {e}')
        raise FileNotFoundError(e)


def export_to_onnx(weights, img, model, dynamic, simplify):
    try:
        import onnx
        f = weights.replace('.pt', '.onnx')  # filename
        train = False
        torch.onnx.export(model, img, f, verbose=False, opset_version=12,
                          training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=not train,
                          input_names=['images'],
                          output_names=['output'],
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                        'output': {0: 'batch', 2: 'y', 3: 'x'}  # shape(1,25200,85)
                                        } if dynamic else None)  # 'output': {0: 'batch', 1: 'anchors'}

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
    except Exception as e:
        print(f'export failure: {e}')
        pass


def export_to_core_ml(weights, img):
    try:
        import coremltools as ct
        model = ct.convert(ts, inputs=[ct.ImageType(name='image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
        f = weights.replace('.pt', '.mlmodel')  # filename
        model.save(f)
    except Exception as e:
        pass


@my_app.callback("export_weights")
@sly.timeit
def export_weights(api: sly.Api, task_id, context, state, app_logger):
    weights_path = download_weights(customWeightsPath)
    cwp = os.path.join(Path(customWeightsPath).parents[1], 'opt.yaml')
    configs_path = download_weights(cwp)
    model = attempt_load(weights=weights_path, map_location=device)

    with open(configs_path, 'r') as stream:
        cfgs_loaded = yaml.safe_load(stream)

    if hasattr(model, 'module') and hasattr(model.module, 'img_size'):
        imgsz = model.module.img_size[0]
    elif hasattr(model, 'img_size'):
        imgsz = model.img_size[0]
    elif cfgs_loaded['img_size']:
        imgsz = cfgs_loaded['img_size'][0]
    else:
        sly.logger.warning(f"Image size is not found in model checkpoint. Use default: {image_size}")
        imgsz = image_size

    gs = int(max(model.stride))
    img_size = [check_img_size(x, gs) for x in [imgsz, imgsz]]
    img = torch.zeros(batch_size, 3, *img_size).to(device)
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()
        if isinstance(m, models.common.Conv):
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    model.model[-1].export = not grid
    for _ in range(2):
        y = model(img)

    # @TODO: fix export_to_onnx for cuda:0
    # ========================================================================
    sly.logger.warning(f"Exporting weights to torchScript format in progress..")
    export_to_torch_script(weights_path, img, model)
    sly.logger.warning(f"Exporting weights to ONNX format in progress..")
    export_to_onnx(weights_path, img, model, dynamic=False, simplify=False)
    # export_to_core_ml(weights_path, img)
    # ========================================================================

    process_folder = str(pathlib.Path(weights_path).parents[0])
    remote_path = customWeightsPath
    remote_path_template = str(pathlib.Path(remote_path).parents[0])
    file_id = None
    for file in os.listdir(process_folder):
        file_path = os.path.join(process_folder, file)
        remote_file_path = os.path.join(remote_path_template, file)
        if '.onnx' in file_path or '.mlmodel' in file_path or '.torchscript' in file_path:
            file_info = api.file.upload(team_id=TEAM_ID, src=file_path, dst=remote_file_path)
            if file_id is None:
                file_id = file_info.id
    if file_id is not None:
        api.task.set_output_directory(task_id, file_id, os.path.dirname(customWeightsPath))
    my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": TEAM_ID,
        "context.workspaceId": WORKSPACE_ID,
        "modal.state.weightsPath": customWeightsPath
    })

    my_app.run(initial_events=[{"command": "export_weights"}])


if __name__ == "__main__":
    sly.main_wrapper("main", main)
