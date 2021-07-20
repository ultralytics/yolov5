import supervisely_lib as sly
from supervisely_lib.io.fs import download, file_exists, get_file_name, get_file_name_with_ext
import os
import pathlib
import torch
import torch.nn as nn

import sys
root_source_path = str(pathlib.Path(sys.argv[0]).parents[3])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)

import models
from utils.general import colorstr, check_img_size, check_requirements, file_size, set_logging
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from models.common import Conv, DWConv

my_app = sly.AppService()

TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
TASK_ID = int(os.environ['TASK_ID'])
customWeightsPath = os.environ['modal.state.slyFile']
device = select_device(device='cpu')
image_size = 640
ts = None
batch_size = 1
grid = True


def export_to_torch_script(weights, img, model):
    global ts
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


def download_weights(path2weights):
    remote_path = path2weights
    weights_path = os.path.join(my_app.data_dir, get_file_name_with_ext(remote_path))
    try:
        my_app.public_api.file.download(team_id=TEAM_ID,
                                        remote_path=remote_path,
                                        local_save_path=weights_path)
        return weights_path
    except:
        raise FileNotFoundError('FileNotFoundError')
        return None


@my_app.callback("export_weights")
@sly.timeit
def export_weights(api: sly.Api, task_id, context, state, app_logger):
    weights_path = download_weights(customWeightsPath)
    model = attempt_load(weights=weights_path, map_location=device)

    if hasattr(model, 'module') and hasattr(model.module, 'img_size'):
        imgsz = model.module.img_size[0]
    elif hasattr(model, 'img_size'):
        imgsz = model.img_size[0]
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
    export_to_torch_script(weights_path, img, model)
    export_to_onnx(weights_path, img, model, dynamic=False, simplify=False)
    # export_to_core_ml(weights_path, img)
    # ========================================================================

    process_folder = str(pathlib.Path(weights_path).parents[0])
    remote_path = customWeightsPath
    remote_path_template = str(pathlib.Path(remote_path).parents[0])
    for file in os.listdir(process_folder):
        file_path = os.path.join(process_folder, file)
        remote_file_path = os.path.join(remote_path_template, file)
        if '.onnx' in file_path or '.mlmodel' in file_path or '.torchscript' in file_path:
            api.file.upload(team_id=TEAM_ID, src=file_path, dst=remote_file_path)
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
