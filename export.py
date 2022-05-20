# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Export a YOLOv5 PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit

Format                      | `export.py --include`         | Model
---                         | ---                           | ---
PyTorch                     | -                             | yolov5s.pt
TorchScript                 | `torchscript`                 | yolov5s.torchscript
ONNX                        | `onnx`                        | yolov5s.onnx
OpenVINO                    | `openvino`                    | yolov5s_openvino_model/
TensorRT                    | `engine`                      | yolov5s.engine
CoreML                      | `coreml`                      | yolov5s.mlmodel
TensorFlow SavedModel       | `saved_model`                 | yolov5s_saved_model/
TensorFlow GraphDef         | `pb`                          | yolov5s.pb
TensorFlow Lite             | `tflite`                      | yolov5s.tflite
TensorFlow Edge TPU         | `edgetpu`                     | yolov5s_edgetpu.tflite
TensorFlow.js               | `tfjs`                        | yolov5s_web_model/

Requirements:
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime openvino-dev tensorflow-cpu  # CPU
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime-gpu openvino-dev tensorflow  # GPU

Usage:
    $ python path/to/export.py --weights yolov5s.pt --include torchscript onnx openvino engine coreml tflite ...

Inference:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolov5/yolov5s_web_model public/yolov5s_web_model
    $ npm start
"""

import argparse
from copy import deepcopy
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

from sparseml.pytorch.utils import ModuleExporter
from sparseml.pytorch.sparsification.quantization import skip_onnx_input_quantize

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import Conv, DetectMultiBackend
from models.experimental import attempt_load
from models.yolo import Detect, Model
from utils.activations import SiLU
from utils.datasets import LoadImages
from utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_version, colorstr,
                           file_size, print_args, url2file, intersect_dicts)
from utils.torch_utils import select_device, torch_distributed_zero_first, is_parallel
from utils.downloads import attempt_download
from utils.sparse import SparseMLWrapper, check_download_sparsezoo_weights



def export_formats():
    # YOLOv5 export formats
    x = [['PyTorch', '-', '.pt', True],
         ['TorchScript', 'torchscript', '.torchscript', True],
         ['ONNX', 'onnx', '.onnx', True],
         ['OpenVINO', 'openvino', '_openvino_model', False],
         ['TensorRT', 'engine', '.engine', True],
         ['CoreML', 'coreml', '.mlmodel', False],
         ['TensorFlow SavedModel', 'saved_model', '_saved_model', True],
         ['TensorFlow GraphDef', 'pb', '.pb', True],
         ['TensorFlow Lite', 'tflite', '.tflite', False],
         ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', False],
         ['TensorFlow.js', 'tfjs', '_web_model', False]]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'GPU'])


def export_torchscript(model, im, file, optimize, prefix=colorstr('TorchScript:')):
    # YOLOv5 TorchScript model export
    try:
        LOGGER.info(f'\n{prefix} starting export with torch {torch.__version__}...')
        f = file.with_suffix('.torchscript')

        ts = torch.jit.trace(model, im, strict=False)
        d = {"shape": im.shape, "stride": int(max(model.stride)), "names": model.names}
        extra_files = {'config.txt': json.dumps(d)}  # torch._C.ExtraFilesMap()
        if optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
            optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
        else:
            ts.save(str(f), _extra_files=extra_files)

        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'{prefix} export failure: {e}')


def export_onnx(model, im, file, opset, train, dynamic, simplify, prefix=colorstr('ONNX:')):
    # YOLOv5 ONNX export
    try:
        check_requirements(('onnx',))
        import onnx

        LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        f = file.with_suffix('.onnx')

        # export through SparseML so quantized and pruned graphs can be corrected
        save_dir = f.parent.absolute()
        save_name = str(f).split(os.path.sep)[-1]

        # get the number of outputs so we know how to name and change dynamic axes
        # nested outputs can be returned if model is exported with dynamic
        def _count_outputs(outputs):
            count = 0
            if isinstance(outputs, list) or isinstance(outputs, tuple):
                for out in outputs:
                    count += _count_outputs(out)
            else:
                count += 1
            return count

        outputs = model(im)
        num_outputs = _count_outputs(outputs)
        input_names = ['input']
        output_names = [f'out_{i}' for i in range(num_outputs)]
        dynamic_axes = {k: {0: 'batch'} for k in (input_names + output_names)} if dynamic else None
        exporter = ModuleExporter(model, save_dir)
        exporter.export_onnx(im, name=save_name, convert_qat=True,
                                input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)
        try:
            skip_onnx_input_quantize(f, f)
        except:
            pass

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # LOGGER.info(onnx.helper.printable_graph(model_onnx.graph))  # print

        # Simplify
        if simplify:
            try:
                check_requirements(('onnx-simplifier',))
                import onnxsim

                LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={'images': list(im.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                LOGGER.info(f'{prefix} simplifier failure: {e}')
        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'{prefix} export failure: {e}')


def export_openvino(model, im, file, prefix=colorstr('OpenVINO:')):
    # YOLOv5 OpenVINO export
    try:
        check_requirements(('openvino-dev',))  # requires openvino-dev: https://pypi.org/project/openvino-dev/
        import openvino.inference_engine as ie

        LOGGER.info(f'\n{prefix} starting export with openvino {ie.__version__}...')
        f = str(file).replace('.pt', '_openvino_model' + os.sep)

        cmd = f"mo --input_model {file.with_suffix('.onnx')} --output_dir {f}"
        subprocess.check_output(cmd, shell=True)

        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')


def export_coreml(model, im, file, prefix=colorstr('CoreML:')):
    # YOLOv5 CoreML export
    try:
        check_requirements(('coremltools',))
        import coremltools as ct

        LOGGER.info(f'\n{prefix} starting export with coremltools {ct.__version__}...')
        f = file.with_suffix('.mlmodel')

        ts = torch.jit.trace(model, im, strict=False)  # TorchScript model
        ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=im.shape, scale=1 / 255, bias=[0, 0, 0])])
        ct_model.save(f)

        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return ct_model, f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')
        return None, None


def export_engine(model, im, file, train, half, simplify, workspace=4, verbose=False, prefix=colorstr('TensorRT:')):
    # YOLOv5 TensorRT export https://developer.nvidia.com/tensorrt
    try:
        check_requirements(('tensorrt',))
        import tensorrt as trt

        if trt.__version__[0] == '7':  # TensorRT 7 handling https://github.com/ultralytics/yolov5/issues/6012
            grid = model.model[-1].anchor_grid
            model.model[-1].anchor_grid = [a[..., :1, :1, :] for a in grid]
            export_onnx(model, im, file, 12, train, False, simplify)  # opset 12
            model.model[-1].anchor_grid = grid
        else:  # TensorRT >= 8
            check_version(trt.__version__, '8.0.0', hard=True)  # require tensorrt>=8.0.0
            export_onnx(model, im, file, 13, train, False, simplify)  # opset 13
        onnx = file.with_suffix('.onnx')

        LOGGER.info(f'\n{prefix} starting export with TensorRT {trt.__version__}...')
        assert im.device.type != 'cpu', 'export running on CPU but must be on GPU, i.e. `python export.py --device 0`'
        assert onnx.exists(), f'failed to export ONNX file: {onnx}'
        f = file.with_suffix('.engine')  # TensorRT engine file
        logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = workspace * 1 << 30
        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError(f'failed to load ONNX file: {onnx}')

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        LOGGER.info(f'{prefix} Network Description:')
        for inp in inputs:
            LOGGER.info(f'{prefix}\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
        for out in outputs:
            LOGGER.info(f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')

        LOGGER.info(f'{prefix} building FP{16 if builder.platform_has_fast_fp16 else 32} engine in {f}')
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
            t.write(engine.serialize())
        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')


def export_saved_model(model, im, file, dynamic,
                       tf_nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45,
                       conf_thres=0.25, keras=False, prefix=colorstr('TensorFlow SavedModel:')):
    # YOLOv5 TensorFlow SavedModel export
    try:
        import tensorflow as tf
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

        from models.tf import TFDetect, TFModel

        LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        f = str(file).replace('.pt', '_saved_model')
        batch_size, ch, *imgsz = list(im.shape)  # BCHW

        tf_model = TFModel(cfg=model.yaml, model=model, nc=model.nc, imgsz=imgsz)
        im = tf.zeros((batch_size, *imgsz, ch))  # BHWC order for TensorFlow
        _ = tf_model.predict(im, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
        inputs = tf.keras.Input(shape=(*imgsz, ch), batch_size=None if dynamic else batch_size)
        outputs = tf_model.predict(inputs, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
        keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        keras_model.trainable = False
        keras_model.summary()
        if keras:
            keras_model.save(f, save_format='tf')
        else:
            m = tf.function(lambda x: keras_model(x))  # full model
            spec = tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype)
            m = m.get_concrete_function(spec)
            frozen_func = convert_variables_to_constants_v2(m)
            tfm = tf.Module()
            tfm.__call__ = tf.function(lambda x: frozen_func(x)[0], [spec])
            tfm.__call__(im)
            tf.saved_model.save(
                tfm,
                f,
                options=tf.saved_model.SaveOptions(experimental_custom_gradients=False) if
                check_version(tf.__version__, '2.6') else tf.saved_model.SaveOptions())
        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return keras_model, f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')
        return None, None


def export_pb(keras_model, im, file, prefix=colorstr('TensorFlow GraphDef:')):
    # YOLOv5 TensorFlow GraphDef *.pb export https://github.com/leimao/Frozen_Graph_TensorFlow
    try:
        import tensorflow as tf
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

        LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        f = file.with_suffix('.pb')

        m = tf.function(lambda x: keras_model(x))  # full model
        m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
        frozen_func = convert_variables_to_constants_v2(m)
        frozen_func.graph.as_graph_def()
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(f.parent), name=f.name, as_text=False)

        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')


def export_tflite(keras_model, im, file, int8, data, ncalib, prefix=colorstr('TensorFlow Lite:')):
    # YOLOv5 TensorFlow Lite export
    try:
        import tensorflow as tf

        LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        batch_size, ch, *imgsz = list(im.shape)  # BCHW
        f = str(file).replace('.pt', '-fp16.tflite')

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.target_spec.supported_types = [tf.float16]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if int8:
            from models.tf import representative_dataset_gen
            dataset = LoadImages(check_dataset(data)['train'], img_size=imgsz, auto=False)  # representative data
            converter.representative_dataset = lambda: representative_dataset_gen(dataset, ncalib)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.target_spec.supported_types = []
            converter.inference_input_type = tf.uint8  # or tf.int8
            converter.inference_output_type = tf.uint8  # or tf.int8
            converter.experimental_new_quantizer = True
            f = str(file).replace('.pt', '-int8.tflite')

        tflite_model = converter.convert()
        open(f, "wb").write(tflite_model)
        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')


def export_edgetpu(keras_model, im, file, prefix=colorstr('Edge TPU:')):
    # YOLOv5 Edge TPU export https://coral.ai/docs/edgetpu/models-intro/
    try:
        cmd = 'edgetpu_compiler --version'
        help_url = 'https://coral.ai/docs/edgetpu/compiler/'
        assert platform.system() == 'Linux', f'export only supported on Linux. See {help_url}'
        if subprocess.run(cmd + ' >/dev/null', shell=True).returncode != 0:
            LOGGER.info(f'\n{prefix} export requires Edge TPU compiler. Attempting install from {help_url}')
            sudo = subprocess.run('sudo --version >/dev/null', shell=True).returncode == 0  # sudo installed on system
            for c in ['curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -',
                      'echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list',
                      'sudo apt-get update',
                      'sudo apt-get install edgetpu-compiler']:
                subprocess.run(c if sudo else c.replace('sudo ', ''), shell=True, check=True)
        ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1]

        LOGGER.info(f'\n{prefix} starting export with Edge TPU compiler {ver}...')
        f = str(file).replace('.pt', '-int8_edgetpu.tflite')  # Edge TPU model
        f_tfl = str(file).replace('.pt', '-int8.tflite')  # TFLite model

        cmd = f"edgetpu_compiler -s {f_tfl}"
        subprocess.run(cmd, shell=True, check=True)

        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')


def export_tfjs(keras_model, im, file, prefix=colorstr('TensorFlow.js:')):
    # YOLOv5 TensorFlow.js export
    try:
        check_requirements(('tensorflowjs',))
        import re

        import tensorflowjs as tfjs

        LOGGER.info(f'\n{prefix} starting export with tensorflowjs {tfjs.__version__}...')
        f = str(file).replace('.pt', '_web_model')  # js dir
        f_pb = file.with_suffix('.pb')  # *.pb path
        f_json = f + '/model.json'  # *.json path

        cmd = f'tensorflowjs_converter --input_format=tf_frozen_model ' \
              f'--output_node_names="Identity,Identity_1,Identity_2,Identity_3" {f_pb} {f}'
        subprocess.run(cmd, shell=True)

        json = open(f_json).read()
        with open(f_json, 'w') as j:  # sort JSON Identity_* in ascending order
            subst = re.sub(
                r'{"outputs": {"Identity.?.?": {"name": "Identity.?.?"}, '
                r'"Identity.?.?": {"name": "Identity.?.?"}, '
                r'"Identity.?.?": {"name": "Identity.?.?"}, '
                r'"Identity.?.?": {"name": "Identity.?.?"}}}',
                r'{"outputs": {"Identity": {"name": "Identity"}, '
                r'"Identity_1": {"name": "Identity_1"}, '
                r'"Identity_2": {"name": "Identity_2"}, '
                r'"Identity_3": {"name": "Identity_3"}}}',
                json)
            j.write(subst)

        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')

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
            'hyp': model.hyp,
            **ema.state_dict(pickle),
            **sparseml_wrapper.state_dict(),
            **kwargs}

def load_checkpoint(
    type_, 
    weights, 
    device, 
    cfg=None, 
    hyp=None, 
    nc=None, 
    data=None, 
    dnn=False, 
    half = False, 
    recipe=None, 
    resume=None, 
    rank=-1
    ):
    with torch_distributed_zero_first(rank):
        # download if not found locally or from sparsezoo if stub
        weights = attempt_download(weights) or check_download_sparsezoo_weights(weights)
    ckpt = torch.load(weights[0] if isinstance(weights, list) or isinstance(weights, tuple)
                      else weights, map_location="cpu")  # load checkpoint
    start_epoch = ckpt['epoch'] + 1 if 'epoch' in ckpt else 0
    pickled = isinstance(ckpt['model'], nn.Module)
    train_type = type_ == 'train'
    ensemble_type = type_ == 'ensemble'
    val_type = type_ =='val'

    if pickled and ensemble_type:
        cfg = None
        if ensemble_type:
            model = attempt_load(weights, map_location=device) # load ensemble using pickled
            state_dict = model.state_dict()
        elif val_type:
            model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
            state_dict = model.model.state_dict()
    else:
        # load model from config and weights
        cfg = cfg or (ckpt['yaml'] if 'yaml' in ckpt else None) or \
              (ckpt['model'].yaml if pickled else None)
        model = Model(cfg, ch=3, nc=ckpt['nc'] if ('nc' in ckpt and not nc) else nc,
                      anchors=hyp.get('anchors') if hyp else None).to(device)
        model_key = 'ema' if (not train_type and 'ema' in ckpt and ckpt['ema']) else 'model'
        state_dict = ckpt[model_key].float().state_dict() if pickled else ckpt[model_key]
        if val_type:
            model = DetectMultiBackend(model=model, device=device, dnn=dnn, data=data, fp16=half)

    # turn gradients for params back on in case they were removed
    for p in model.parameters():
        p.requires_grad = True

    # load sparseml recipe for applying pruning and quantization
    checkpoint_recipe = train_recipe = None
    if resume:
        train_recipe = ckpt.get('recipe')
    elif recipe or ckpt.get('recipe'):
        train_recipe, checkpoint_recipe = recipe, ckpt.get('recipe')

    sparseml_wrapper = SparseMLWrapper(model.model if val_type else model, checkpoint_recipe, train_recipe)
    exclude_anchors = train_type and (cfg or hyp.get('anchors')) and not resume
    loaded = False

    sparseml_wrapper.apply_checkpoint_structure()
    if train_type:
        # intialize the recipe for training and restore the weights before if no quantized weights
        quantized_state_dict = any([name.endswith('.zero_point') for name in state_dict.keys()])
        if not quantized_state_dict:
            state_dict = load_state_dict(model, state_dict, train=True, exclude_anchors=exclude_anchors)
            loaded = True
        sparseml_wrapper.initialize(start_epoch)

    if not loaded:
        state_dict = load_state_dict(model, state_dict, train=train_type, exclude_anchors=exclude_anchors)

    model.float()
    report = 'Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights)

    if val_type:
        model.model.eval()
        
    return model, {
        'ckpt': ckpt,
        'state_dict': state_dict,
        'sparseml_wrapper': sparseml_wrapper,
        'report': report,
    }


def load_state_dict(model, state_dict, train, exclude_anchors):
    # fix older state_dict names not porting to the new model setup
    state_dict = {key if not key.startswith("module.") else key[7:]: val for key, val in state_dict.items()}

    if train:
        # load any missing weights from the model
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=['anchor'] if exclude_anchors else [])

    model.load_state_dict(state_dict, strict=not train)  # load

    return state_dict

@torch.no_grad()
def run(data=ROOT / 'data/coco128.yaml',  # 'dataset.yaml path'
        weights=ROOT / 'yolov5s.pt',  # weights path
        imgsz=(640, 640),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        include=('onnx'),  # include formats
        half=False,  # FP16 half-precision export
        inplace=False,  # set YOLOv5 Detect() inplace=True
        train=False,  # model.train() mode
        optimize=False,  # TorchScript: optimize for mobile
        int8=False,  # CoreML/TF INT8 quantization
        dynamic=False,  # ONNX/TF: dynamic axes
        simplify=False,  # ONNX: simplify model
        opset=12,  # ONNX: opset version
        verbose=False,  # TensorRT: verbose log
        workspace=4,  # TensorRT: workspace size (GB)
        nms=False,  # TF: add NMS to model
        agnostic_nms=False,  # TF: add agnostic NMS to model
        topk_per_class=100,  # TF.js NMS: topk per class to keep
        topk_all=100,  # TF.js NMS: topk for all classes to keep
        iou_thres=0.45,  # TF.js NMS: IoU threshold
        conf_thres=0.25,  # TF.js NMS: confidence threshold
        remove_grid=False,
        ):
    t = time.time()
    include = [x.lower() for x in include]  # to lowercase
    formats = tuple(export_formats()['Argument'][1:])  # --include arguments
    flags = [x in include for x in formats]
    assert sum(flags) == len(include), f'ERROR: Invalid --include {include}, valid --include arguments are {formats}'
    jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs = flags  # export booleans
    file = Path(url2file(weights) if str(weights).startswith(('http:/', 'https:/')) else weights)  # PyTorch weights

    # Load PyTorch model
    device = select_device(device)
    assert not (device.type == 'cpu' and half), '--half only compatible with GPU export, i.e. use --device 0'
    model, extras = load_checkpoint(type_='ensemble', weights=weights, device=device)  # load FP32 model
    sparseml_wrapper = extras['sparseml_wrapper']
    nc, names = extras["ckpt"].get("nc") or model.nc, model.names  # number of classes, class names

    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    opset = 12 if ('openvino' in include) else opset  # OpenVINO requires opset <= 12
    assert nc == len(names), f'Model class count {nc} != len(names) {len(names)}'

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
            if hasattr(m, 'forward_export'):
                m.forward = m.forward_export  # assign custom forward (optional)
    model.model[-1].export = not remove_grid  # set Detect() layer grid export

    for _ in range(2):
        y = model(im)  # dry runs
    shape = tuple(y[0].shape)  # model output shape
    LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")

    # Exports
    f = [''] * 10  # exported filenames
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
    if jit:
        f[0] = export_torchscript(model, im, file, optimize)
    if engine:  # TensorRT required before ONNX
        f[1] = export_engine(model, im, file, train, half, simplify, workspace, verbose)
    if onnx or xml:  # OpenVINO requires ONNX
        f[2] = export_onnx(model, im, file, opset, train, dynamic, simplify)
    if xml:  # OpenVINO
        f[3] = export_openvino(model, im, file)
    if coreml:
        _, f[4] = export_coreml(model, im, file)

    # TensorFlow Exports
    if any((saved_model, pb, tflite, edgetpu, tfjs)):
        if int8 or edgetpu:  # TFLite --int8 bug https://github.com/ultralytics/yolov5/issues/5707
            check_requirements(('flatbuffers==1.12',))  # required before `import tensorflow`
        assert not (tflite and tfjs), 'TFLite and TF.js models must be exported separately, please pass only one type.'
        model, f[5] = export_saved_model(model.cpu(), im, file, dynamic, tf_nms=nms or agnostic_nms or tfjs,
                                         agnostic_nms=agnostic_nms or tfjs, topk_per_class=topk_per_class,
                                         topk_all=topk_all, conf_thres=conf_thres, iou_thres=iou_thres)  # keras model
        if pb or tfjs:  # pb prerequisite to tfjs
            f[6] = export_pb(model, im, file)
        if tflite or edgetpu:
            f[7] = export_tflite(model, im, file, int8=int8 or edgetpu, data=data, ncalib=100)
        if edgetpu:
            f[8] = export_edgetpu(model, im, file)
        if tfjs:
            f[9] = export_tfjs(model, im, file)

    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        LOGGER.info(f'\nExport complete ({time.time() - t:.2f}s)'
                    f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                    f"\nDetect:          python detect.py --weights {f[-1]}"
                    f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{f[-1]}')"
                    f"\nValidate:        python val.py --weights {f[-1]}"
                    f"\nVisualize:       https://netron.app")
    return f  # return list of exported files/dirs


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--train', action='store_true', help='model.train() mode')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
    parser.add_argument('--int8', action='store_true', help='CoreML/TF INT8 quantization')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TF: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
    parser.add_argument('--verbose', action='store_true', help='TensorRT: verbose log')
    parser.add_argument('--workspace', type=int, default=4, help='TensorRT: workspace size (GB)')
    parser.add_argument('--nms', action='store_true', help='TF: add NMS to model')
    parser.add_argument('--agnostic-nms', action='store_true', help='TF: add agnostic NMS to model')
    parser.add_argument('--topk-per-class', type=int, default=100, help='TF.js NMS: topk per class to keep')
    parser.add_argument('--topk-all', type=int, default=100, help='TF.js NMS: topk for all classes to keep')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='TF.js NMS: IoU threshold')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='TF.js NMS: confidence threshold')
    parser.add_argument("--remove-grid", action="store_true", help="remove export of Detect() layer grid")
    parser.add_argument('--include', nargs='+',
                        default=['torchscript', 'onnx'],
                        help='torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs')
    opt = parser.parse_args()
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    for opt.weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)