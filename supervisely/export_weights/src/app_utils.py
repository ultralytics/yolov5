import numpy as np
import onnxruntime as rt
import os
import torch
import yaml
# import sys
# from pathlib import Path
# sys.path.append(Path(sys.argv[0]))
from models.experimental import attempt_load
from PIL import Image
from supervisely.io.fs import get_file_name_with_ext
from torchvision import transforms
from sly_globals import my_app, TEAM_ID


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def download_file(path2file, app, team_id):
    assert app, 'App object should be passed'
    assert team_id, 'team_id should be passed'
    remote_path = path2file
    local_file_path = os.path.join(app.data_dir, get_file_name_with_ext(remote_path))
    if os.path.exists(local_file_path):
        return local_file_path
    try:
        app.public_api.file.download(team_id=team_id,
                                     remote_path=remote_path,
                                     local_save_path=local_file_path)
        return local_file_path
    except:
        raise FileNotFoundError('FileNotFoundError')


def get_image(path2image):
    _image_ = Image.open(path2image)
    _tensor = transforms.PILToTensor()(_image_)
    _tensor = _tensor.unsqueeze(0) / 255
    return _tensor, _image_


def onnx_inference(path_to_saved_model):
    onnx_model_ = rt.InferenceSession(path_to_saved_model)
    input_name = onnx_model_.get_inputs()[0].name
    label_name = onnx_model_.get_outputs()[0].name
    return onnx_model_, input_name, label_name


def get_model(path2model,
              app=None, _team_id=None):
    if not os.path.exists(path2model):
        assert app, 'app object should be passed'
        assert _team_id, 'TEAM_ID should be passed'
        try:
            path_to_saved_model = download_file(path2file=path2model,
                                                app=app,
                                                team_id=_team_id)
        except:
            raise FileNotFoundError(path2model)
    else:
        path_to_saved_model = path2model

    if 'pt' in path_to_saved_model:
        if 'torchscript' in path_to_saved_model:
            _model = torch.jit.load(path_to_saved_model)
        else:
            _model = attempt_load(weights=path_to_saved_model)  # , map_location=device
    if 'onnx' in path_to_saved_model:
        _model = onnx_inference(path_to_saved_model)
    return _model


def get_configs(cfgs_path, app=None, _team_id=None):
    if not os.path.exists(cfgs_path):
        if app and _team_id:
            cfgs_path = download_file(path2file=cfgs_path, app=app, team_id=_team_id)
        else:
            raise FileNotFoundError(cfgs_path)

    with open(cfgs_path, 'r') as yaml_file:
        cfgs = yaml.load(yaml_file)
    return cfgs


def preprocess(predictions, original_image_size, reshaped_image_size):
    coefficients = torch.tensor(original_image_size) / torch.tensor(reshaped_image_size)
    predictions_copy = predictions
    for item in predictions_copy:
        item[..., 0] *= coefficients[0]
        item[..., 1] *= coefficients[1]
        item[..., 2] *= coefficients[0]
        item[..., 3] *= coefficients[1]
    return predictions_copy


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
