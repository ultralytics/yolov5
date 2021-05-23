"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights yolov5s.pt --img 640 --batch 1
"""

import argparse
from copy import deepcopy
import sys
import time

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn

from sparseml.pytorch.utils import ModuleExporter
from sparseml.pytorch.utils.quantization import skip_onnx_input_quantize

import models
from models.experimental import attempt_load
from models.yolo import Model
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.google_utils import attempt_download
from utils.sparse import SparseMLWrapper
from utils.torch_utils import select_device, intersect_dicts, is_parallel, torch_distributed_zero_first


def create_checkpoint(epoch, model, optimizer, ema, sparseml_wrapper, **kwargs):
    pickle = not sparseml_wrapper.qat_active(epoch)  # qat does not support pickled exports
    ckpt_model = deepcopy(model.module if is_parallel(model) else model).float()
    yaml = ckpt_model.yaml
    if not pickle:
        ckpt_model = ckpt_model.state_dict()

    return {'epoch': epoch,
            'model': ckpt_model,
            'optimizer': optimizer.state_dict(),
            'yaml': yaml,
            **ema.state_dict(pickle),
            **sparseml_wrapper.state_dict(),
            **kwargs}


def load_checkpoint(type_, weights, device, cfg=None, hyp=None, nc=None, recipe=None, resume=None, rank=-1):
    with torch_distributed_zero_first(rank):
        attempt_download(weights)  # download if not found locally
    ckpt = torch.load(weights, map_location=device)  # load checkpoint
    start_epoch = ckpt['epoch'] + 1 if 'epoch' in ckpt else 0
    pickled = isinstance(ckpt['model'], nn.Module)

    if pickled and type_ == 'ensemble':
        # load ensemble using pickled
        cfg = None
        model = attempt_load(weights, map_location=device)  # load FP32 model
        state_dict = model.state_dict()
    else:
        # load model from config and weights
        cfg = cfg or (ckpt['yaml'] if 'yaml' in ckpt else None) or \
              (ckpt['model'].yaml if pickled else None)
        model = Model(cfg, ch=3, nc=ckpt['nc'] if ('nc' in ckpt and not nc) else nc,
                      anchors=hyp.get('anchors') if hyp else None).to(device)
        model_key = 'ema' if (type_ in ['ema', 'ensemble'] and 'ema' in ckpt and ckpt['ema']) else 'model'
        state_dict = ckpt[model_key].float().state_dict() if pickled else ckpt[model_key]

    # turn gradients for params back on in case they were removed
    for p in model.parameters():
        p.requires_grad = True

    # load sparseml recipe for applying pruning and quantization
    recipe = recipe or (ckpt['recipe'] if 'recipe' in ckpt else None)
    sparseml_wrapper = SparseMLWrapper(model, recipe)
    if type_ in ['ema', 'ensemble']:
        # apply the recipe to create the final state of the model when not training
        sparseml_wrapper.apply()
    else:
        # intialize the recipe for training
        sparseml_wrapper.initialize(start_epoch)

    if type_ == 'train':
        # load any missing weights from the model
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect

    model.load_state_dict(state_dict, strict=type_ != 'train')  # load
    model.float()
    report = 'Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights)

    return model, {
        'ckpt': ckpt,
        'state_dict': state_dict,
        'start_epoch': start_epoch,
        'sparseml_wrapper': sparseml_wrapper,
        'report': report,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov3.pt', help='weights path')  # from yolov3/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.device)
    model, extras = load_checkpoint('ensemble', opt.weights, device)  # load FP32 model
    sparseml_wrapper = extras['sparseml_wrapper']
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = not opt.grid  # set Detect() layer grid export
    y = model(img)  # dry run

    # TorchScript export
    try:
        print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
        ts = torch.jit.trace(model, img, strict=False)
        ts.save(f)
        print('TorchScript export success, saved as %s' % f)
    except Exception as e:
        print('TorchScript export failure: %s' % e)

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        if not sparseml_wrapper.enabled:
            torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                              output_names=['classes', 'boxes'] if y is None else ['output'],
                              dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                                            'output': {0: 'batch', 2: 'y', 3: 'x'}} if opt.dynamic else None)
        else:
            # export through SparseML so quantized and pruned graphs can be corrected
            save_dir = '/'.join(f.split('/')[:-1])
            save_name = f.split('/')[-1]
            exporter = ModuleExporter(model, save_dir)
            exporter.export_onnx(img, name=save_name, convert_qat=True)
            try:
                skip_onnx_input_quantize(f, f)
            except:
                pass

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # CoreML export
    try:
        import coremltools as ct

        print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
        # convert model from torchscript and apply pixel scaling as per detect.py
        model = ct.convert(ts, inputs=[ct.ImageType(name='image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
        f = opt.weights.replace('.pt', '.mlmodel')  # filename
        model.save(f)
        print('CoreML export success, saved as %s' % f)
    except Exception as e:
        print('CoreML export failure: %s' % e)

    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))