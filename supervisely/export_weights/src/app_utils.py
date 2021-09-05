import torch
import numpy as np
import os
from supervisely_lib.io.fs import get_file_name_with_ext


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def download_weights(path2weights, **kwargs):
    remote_path = path2weights
    weights_path = os.path.join(kwargs['my_app'].data_dir, get_file_name_with_ext(remote_path))
    if os.path.exists(weights_path):
        return weights_path
    try:
        kwargs['my_app'].public_api.file.download(team_id=kwargs['TEAM_ID'],
                                                  remote_path=remote_path,
                                                  local_save_path=weights_path)
        return weights_path
    except:
        raise FileNotFoundError('FileNotFoundError')
        return None
