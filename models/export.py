"""Export a YOLOv5 *.pt model to TorchScript, ONNX, CoreML formats

Usage:
    $ python path/to/models/export.py --weights yolov5s.pt --img 640 --batch 1
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add yolov5/ to path

from models.common import Conv
from models.yolo import Detect
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import colorstr, check_img_size, check_requirements, file_size, set_logging
from utils.torch_utils import select_device


def export(weights='./yolov5s.pt',  # weights path
           img_size=(640, 640),  # image (height, width)
           batch_size=1,  # batch size
           device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           include=('torchscript', 'onnx', 'coreml'),  # include formats
           half=False,  # FP16 half-precision export
           inplace=False,  # set YOLOv5 Detect() inplace=True
           train=False,  # model.train() mode
           optimize=False,  # TorchScript: optimize for mobile
           dynamic=False,  # ONNX: dynamic axes
           simplify=False,  # ONNX: simplify model
           opset_version=12,  # ONNX: opset version
           ):
    t = time.time()
    include = [x.lower() for x in include]
    img_size *= 2 if len(img_size) == 1 else 1  # expand

    # Load PyTorch model
    device = select_device(device)
    assert not (device.type == 'cpu' and opt.half), '--half only compatible with GPU export, i.e. use --device 0'
    model = attempt_load(weights, map_location=device)  # load FP32 model
    labels = model.names

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    img_size = [check_img_size(x, gs) for x in img_size]  # verify img_size are gs-multiples
    img = torch.zeros(batch_size, 3, *img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    if half:
        img, model = img.half(), model.half()  # to FP16
    model.train() if train else model.eval()  # training mode = no Detect() layer grid construction
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = inplace
            m.onnx_dynamic = dynamic
            # m.forward = m.forward_export  # assign forward (optional)

    for _ in range(2):
        y = model(img)  # dry runs
    print(f"\n{colorstr('PyTorch:')} starting from {weights} ({file_size(weights):.1f} MB)")

    # TorchScript export -----------------------------------------------------------------------------------------------
    if 'torchscript' in include or 'coreml' in include:
        prefix = colorstr('TorchScript:')
        try:
            print(f'\n{prefix} starting export with torch {torch.__version__}...')
            f = weights.replace('.pt', '.torchscript.pt')  # filename
            ts = torch.jit.trace(model, img, strict=False)
            (optimize_for_mobile(ts) if optimize else ts).save(f)
            print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        except Exception as e:
            print(f'{prefix} export failure: {e}')

    # ONNX export ------------------------------------------------------------------------------------------------------
    if 'onnx' in include:
        prefix = colorstr('ONNX:')
        try:
            import onnx

            print(f'{prefix} starting export with onnx {onnx.__version__}...')
            f = weights.replace('.pt', '.onnx')  # filename
            torch.onnx.export(model, img, f, verbose=False, opset_version=opset_version,
                              training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                              do_constant_folding=not train,
                              input_names=['images'],
                              output_names=['output'],
                              dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                            'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
                                            } if dynamic else None)

            # Checks
            model_onnx = onnx.load(f)  # load onnx model
            onnx.checker.check_model(model_onnx)  # check onnx model
            # print(onnx.helper.printable_graph(model_onnx.graph))  # print

            # Simplify
            if simplify:
                try:
                    check_requirements(['onnx-simplifier'])
                    import onnxsim

                    print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                    model_onnx, check = onnxsim.simplify(
                        model_onnx,
                        dynamic_input_shape=dynamic,
                        input_shapes={'images': list(img.shape)} if dynamic else None)
                    assert check, 'assert check failed'
                    onnx.save(model_onnx, f)
                except Exception as e:
                    print(f'{prefix} simplifier failure: {e}')
            print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        except Exception as e:
            print(f'{prefix} export failure: {e}')

    # CoreML export ----------------------------------------------------------------------------------------------------
    if 'coreml' in include:
        prefix = colorstr('CoreML:')
        try:
            import coremltools as ct

            print(f'{prefix} starting export with coremltools {ct.__version__}...')
            assert train, 'CoreML exports should be placed in model.train() mode with `python export.py --train`'
            model = ct.convert(ts, inputs=[ct.ImageType('image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
            f = weights.replace('.pt', '.mlmodel')  # filename
            model.save(f)
            print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        except Exception as e:
            print(f'{prefix} export failure: {e}')

    # Finish
    print(f'\nExport complete ({time.time() - t:.2f}s). Visualize with https://github.com/lutzroeder/netron.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image (height, width)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--include', nargs='+', default=['torchscript', 'onnx', 'coreml'], help='include formats')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--train', action='store_true', help='model.train() mode')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
    parser.add_argument('--dynamic', action='store_true', help='ONNX: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset-version', type=int, default=12, help='ONNX: opset version')
    opt = parser.parse_args()
    print(opt)
    set_logging()

    export(**vars(opt))
