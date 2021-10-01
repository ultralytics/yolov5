import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path
import copy

import math

#sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

import torch_pruning as tp
from utils.general import check_file, set_logging
from utils.torch_utils import select_device
from models.common import BottleneckCSP, C3, Concat
from models.yolo import Model

from functools import partial

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


def get_prune_conv_layers(model, out_layer_names=[]):
    in_layers = []
    out_layers = []
    names = []
    for k, v in model.named_modules():
        if isinstance(v, nn.Conv2d):
            if k in out_layer_names:
                out_layers.append(v)
            else:
                in_layers.append(v)
                names.append(k)
    return in_layers, out_layers

def prune_conv(conv, DG, amount=0.5, return_idxs=False):
    strategy = tp.strategy.L1Strategy()
    pruning_index = strategy(conv.weight, amount=amount)
    plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
    plan.exec()
    if return_idxs:
        return pruning_index

#def prune_conv_graphless(conv, amount):
#    strategy = tp.strategy.L1Strategy()
#    pruning_idxs = strategy(conv.weight, amount)
#    tp.prune_conv(conv, idxs=pruning_idxs)


def prune_model(model, out_layers_names, amount=0.5, input_shape=(1, 3, 512, 512)):
    # Prune convolution kernels of a sequential model
    model.cpu()
    in_layers, out_layers = get_prune_conv_layers(model, out_layers_names) # Separate inner and outer layers
    DG = tp.DependencyGraph().build_dependency(model, torch.randn(*input_shape))
    # Prune all inner conv layers
    for layer in in_layers:
        prune_conv(layer, DG, amount)
    out_layers_idxs = []
    # Prune outer conv layers and return indices of kernels that are kept
    for layer in out_layers:
        idxs = prune_conv(layer, DG, amount, return_idxs=True)
        out_layers_idxs.append(idxs)
    return out_layers_idxs

def prune_detect_layer(model, out_layers_idxs):
    out_conv_strategy = tp.strategy.L1Strategy()
    detect_convs = model.model[-1].m
    for i, detect_conv in enumerate(detect_convs):
        tp.prune_related_conv(detect_conv, idxs=out_layers_idxs[i])


def get_last_conv_layer_suffix(module):
    # Return most final convolution layer for each endblock type (BottleneckCSP, C3, etc...)
    # This is useful to save the pruning indices of that layer as to rearrange kernels
    # for the Detect layer's convoluions
    if isinstance(module, BottleneckCSP):
        return 'cv4.conv'
    elif isinstance(module, C3):
        return 'cv3.conv'
    else:
        raise NotImplemented()


def create_sequential_view(model, input_shape=(1, 3, 512, 512)):
    # Create a torch.nn.Sequential view of YOLOv5 for ease of pruning

    def save_activation(activations, name, mod, inp, out):
            activations[name] = out

    activations = dict()
    model_children = list(*model.children())
    concat_idxs = []
    concat_skip_idx = []
    for idx, child in enumerate(model_children):
        if isinstance(child, Concat):
            concat_idxs.append(idx)
            concat_skip_idx.append(child.f[1])
    hooks = []
    for idx in concat_skip_idx:
        hooks.append(
                model_children[idx].register_forward_hook(
                    partial(save_activation, activations, idx)
            )
        )
    sequential_modules = model_children[0:concat_idxs[0]]
    for idx in range(len(concat_idxs) - 1):
        skip_idx = concat_skip_idx[idx]
        sequential_modules += [LambdaLayer(partial(lambda u, x: torch.cat((x, activations[u]), dim=1), concat_skip_idx[idx]))]
        sequential_modules += model_children[concat_idxs[idx]+1:concat_idxs[idx+1]]
    sequential_modules += [LambdaLayer(partial(lambda u, x: torch.cat((x, activations[u]), dim=1), concat_skip_idx[-1]))]
    sequential_modules += model_children[concat_idxs[-1]+1:-1]
    sequential_view = nn.Sequential(*sequential_modules)
    dummy_input = torch.randn(*input_shape)
    # Forward check
    sequential_view(dummy_input)
    # Get output conv layer names
    out_layers_names = []
    for out_layer_idx in model_children[-1].f:
        out_layers_names.append('{}.{}'.format(out_layer_idx, get_last_conv_layer_suffix(model_children[out_layer_idx])))
    return sequential_view, out_layers_names, hooks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='Model config file')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='Model weights')
    parser.add_argument('--out-path', default='./outmodel.pt', help='Output model path')
    parser.add_argument('--prune-amount', default=0.5, type=float, help='Pruning amount')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device('cpu')

    # Create model
    ckpt = torch.load(opt.weights, map_location=device)
    model = Model(opt.cfg).to(device)
    model_state_dict = {k:ckpt['model'].state_dict()[k] for k in model.state_dict().keys()}
    model.load_state_dict(model_state_dict)
    sequential_view, out_layers_names, hooks = create_sequential_view(model)
    # Test if model still runs
    dummy_input = torch.randn(1, 3, 512, 512)
    sequential_view(dummy_input)
    out_layers_idxs = prune_model(sequential_view, out_layers_names, amount=opt.prune_amount)
    sequential_view(dummy_input)
    prune_detect_layer(model, out_layers_idxs)
    model(dummy_input)
    for hook in hooks:
        hook.remove()
    out_ckpt = {'model': model}
    torch.save(out_ckpt, opt.out_path)
