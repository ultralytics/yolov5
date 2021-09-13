# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Export a YOLOv5 PyTorch model to TorchScript, ONNX, CoreML, TensorFlow (saved_model, pb, TFLite, TF.js,) formats
TensorFlow exports authored by https://github.com/zldrobit

Usage:
    $ python path/to/export.py --weights yolov5s.pt --include torchscript onnx coreml saved_model pb tflite tfjs

Inference:
    $ python path/to/detect.py --weights yolov5s.pt
                                         yolov5s.onnx  (must export with --dynamic)
                                         yolov5s_saved_model
                                         yolov5s.pb
                                         yolov5s.tflite

TensorFlow.js:
    $ # Edit yolov5s_web_model/model.json to sort Identity* in ascending order
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolov5/yolov5s_web_model public/yolov5s_web_model
    $ npm start
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5/ dir
sys.path.append(ROOT.as_posix())  # add yolov5/ to path

from models.common import Conv
from models.experimental import attempt_load
from models.yolo import Detect
from utils.activations import SiLU
from utils.datasets import LoadImages
from utils.general import colorstr, check_dataset, check_img_size, check_requirements, file_size, set_logging
from utils.torch_utils import select_device


def export_torchscript(model, im, file, optimize, prefix=colorstr('TorchScript:')):
    # YOLOv5 TorchScript model export
    try:
        print(f'\n{prefix} starting export with torch {torch.__version__}...')
        f = file.with_suffix('.torchscript.pt')

        ts = torch.jit.trace(model, im, strict=False)
        (optimize_for_mobile(ts) if optimize else ts).save(f)

        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'{prefix} export failure: {e}')


def export_onnx(model, im, file, opset, train, dynamic, simplify, prefix=colorstr('ONNX:')):
    # YOLOv5 ONNX export
    try:
        check_requirements(('onnx',))
        import onnx

        print(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        f = file.with_suffix('.onnx')

        torch.onnx.export(model, im, f, verbose=False, opset_version=opset,
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
                check_requirements(('onnx-simplifier',))
                import onnxsim

                print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={'images': list(im.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                print(f'{prefix} simplifier failure: {e}')
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        print(f"{prefix} run --dynamic ONNX model inference with: 'python detect.py --weights {f}'")
    except Exception as e:
        print(f'{prefix} export failure: {e}')


def export_coreml(model, im, file, prefix=colorstr('CoreML:')):
    # YOLOv5 CoreML export
    ct_model = None
    try:
        check_requirements(('coremltools',))
        import coremltools as ct

        print(f'\n{prefix} starting export with coremltools {ct.__version__}...')
        f = file.with_suffix('.mlmodel')

        model.train()  # CoreML exports should be placed in model.train() mode
        ts = torch.jit.trace(model, im, strict=False)  # TorchScript model
        ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=im.shape, scale=1 / 255.0, bias=[0, 0, 0])])
        ct_model.save(f)

        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'\n{prefix} export failure: {e}')

    return ct_model


def export_saved_model(model, im, file, dynamic,
                       tf_nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45,
                       conf_thres=0.25, prefix=colorstr('TensorFlow saved_model:')):
    # YOLOv5 TensorFlow saved_model export
    keras_model = None
    try:
        import tensorflow as tf
        from tensorflow import keras
        from models.tf import TFModel, TFDetect

        print(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        f = str(file).replace('.pt', '_saved_model')
        batch_size, ch, *imgsz = list(im.shape)  # BCHW

        tf_model = TFModel(cfg=model.yaml, model=model, nc=model.nc, imgsz=imgsz)
        im = tf.zeros((batch_size, *imgsz, 3))  # BHWC order for TensorFlow
        y = tf_model.predict(im, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
        inputs = keras.Input(shape=(*imgsz, 3), batch_size=None if dynamic else batch_size)
        outputs = tf_model.predict(inputs, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
        keras_model = keras.Model(inputs=inputs, outputs=outputs)
        keras_model.summary()
        keras_model.save(f, save_format='tf')

        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'\n{prefix} export failure: {e}')

    return keras_model


def export_pb(keras_model, im, file, prefix=colorstr('TensorFlow GraphDef:')):
    # YOLOv5 TensorFlow GraphDef *.pb export https://github.com/leimao/Frozen_Graph_TensorFlow
    try:
        import tensorflow as tf
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

        print(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        f = file.with_suffix('.pb')

        m = tf.function(lambda x: keras_model(x))  # full model
        m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
        frozen_func = convert_variables_to_constants_v2(m)
        frozen_func.graph.as_graph_def()
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(f.parent), name=f.name, as_text=False)

        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'\n{prefix} export failure: {e}')


def export_tflite(keras_model, im, file, tfl_int8, data, ncalib, prefix=colorstr('TensorFlow Lite:')):
    # YOLOv5 TensorFlow Lite export
    try:
        import tensorflow as tf
        from models.tf import representative_dataset_gen

        print(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        batch_size, ch, *imgsz = list(im.shape)  # BCHW
        f = file.with_suffix('.tflite')

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if tfl_int8:
            dataset = LoadImages(check_dataset(data)['train'], img_size=imgsz, auto=False)  # representative data
            converter.representative_dataset = lambda: representative_dataset_gen(dataset, ncalib)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8  # or tf.int8
            converter.inference_output_type = tf.uint8  # or tf.int8
            converter.experimental_new_quantizer = False
            f = str(file).replace('.pt', '-int8.tflite')

        tflite_model = converter.convert()
        open(f, "wb").write(tflite_model)
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')

    except Exception as e:
        print(f'\n{prefix} export failure: {e}')


def export_tfjs(keras_model, im, file, prefix=colorstr('TensorFlow.js:')):
    # YOLOv5 TensorFlow.js export
    try:
        check_requirements(('tensorflowjs',))
        import tensorflowjs as tfjs

        print(f'\n{prefix} starting export with tensorflowjs {tfjs.__version__}...')
        f = str(file).replace('.pt', '_web_model')  # js dir
        f_pb = file.with_suffix('.pb')  # *.pb path

        cmd = f"tensorflowjs_converter --input_format=tf_frozen_model " \
              f"--output_node_names='Identity,Identity_1,Identity_2,Identity_3' {f_pb} {f}"
        subprocess.run(cmd, shell=True)

        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'\n{prefix} export failure: {e}')


@torch.no_grad()
def run(data=ROOT / 'data/coco128.yaml',  # 'dataset.yaml path'
        weights=ROOT / 'yolov5s.pt',  # weights path
        imgsz=(640, 640),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        include=('torchscript', 'onnx', 'coreml'),  # include formats
        half=False,  # FP16 half-precision export
        inplace=False,  # set YOLOv5 Detect() inplace=True
        train=False,  # model.train() mode
        optimize=False,  # TorchScript: optimize for mobile
        dynamic=False,  # ONNX: dynamic axes
        simplify=False,  # ONNX: simplify model
        opset=12,  # ONNX: opset version
        ):
    t = time.time()
    include = [x.lower() for x in include]
    tf_exports = list(x in include for x in ('saved_model', 'pb', 'tflite', 'tfjs'))  # TensorFlow exports
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    file = Path(weights)

    # Load PyTorch model
    device = select_device(device)
    assert not (device.type == 'cpu' and half), '--half only compatible with GPU export, i.e. use --device 0'
    model = attempt_load(weights, map_location=device, inplace=True, fuse=not any(tf_exports))  # load FP32 model
    nc, names = model.nc, model.names  # number of classes, class names

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Update model
    if half:
        im, model = im.half(), model.half()  # to FP16
    model.train() if train else model.eval()  # training mode = no Detect() layer grid construction
    for k, m in model.named_modules():
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = inplace
            m.onnx_dynamic = dynamic
            # m.forward = m.forward_export  # assign forward (optional)

    for _ in range(2):
        y = model(im)  # dry runs
    print(f"\n{colorstr('PyTorch:')} starting from {weights} ({file_size(weights):.1f} MB)")

    # Exports
    if 'torchscript' in include:
        export_torchscript(model, im, file, optimize)
    if 'onnx' in include:
        export_onnx(model, im, file, opset, train, dynamic, simplify)
    if 'coreml' in include:
        export_coreml(model, im, file)

    # TensorFlow Exports
    if any(tf_exports):
        pb, tflite, tfjs = tf_exports[1:]
        assert not (tflite and tfjs), 'TFLite and TF.js models must be exported separately, please pass only one type.'
        model = export_saved_model(model, im, file, dynamic, tf_nms=tfjs, agnostic_nms=tfjs)  # keras model
        if pb or tfjs:  # pb prerequisite to tfjs
            export_pb(model, im, file)
        if tflite:
            export_tflite(model, im, file, tfl_int8=False, data=data, ncalib=100)
        if tfjs:
            export_tfjs(model, im, file)

    # Finish
    print(f'\nExport complete ({time.time() - t:.2f}s)'
          f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
          f'\nVisualize with https://netron.app')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='weights path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--train', action='store_true', help='model.train() mode')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TF: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=13, help='ONNX: opset version')
    parser.add_argument('--include', nargs='+',
                        default=['torchscript', 'onnx'],
                        help='available formats are (torchscript, onnx, coreml, saved_model, pb, tflite, tfjs)')
    opt = parser.parse_args()
    return opt


def main(opt):
    set_logging()
    print(colorstr('export: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
