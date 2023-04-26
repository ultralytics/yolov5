#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import argparse
import yaml
import warnings
import collections

import torch
import torch.utils.data
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from prettytable import PrettyTable

import logging
logging.basicConfig(level=logging.ERROR)

try:
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import calib
    from pytorch_quantization.tensor_quant import QuantDescriptor
    from pytorch_quantization import quant_modules
except ImportError:
    raise ImportError(
        "pytorch-quantization is not installed. Install from "
        "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization."
    )

from train import train
import test
from utils_quant.check_params import check_and_set_params
from utils.datasets import create_dataloader
from utils.general import check_img_size, colorstr
from utils.torch_utils import intersect_dicts
from models.yolo import Model


def get_parser():
    """
    Creates an argument parser.
    """
    parser = argparse.ArgumentParser(description='Object detection: Yolov5 quantization flow script')

    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path', required=True)
    parser.add_argument('--out-dir', '-o', default='./runs/finetune', help='output folder: default ./runs/finetune')
    parser.add_argument('--print-freq', '-pf', type=int, default=20, help='evaluation print frequency: default 20')
    parser.add_argument('--threshold', '-t', type=float, default=-1.0, help='top1 accuracy threshold (less than 0.0 means no comparison): default -1.0')

    # setting for yolov5
    parser.add_argument('--model-name', '-m', default='yolov5s', help='model name: default yolov5s')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--ckpt-path', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--batch-size-train', type=int, default=64, help='batch size for training: default 64')
    parser.add_argument('--batch-size-test', type=int, default=64, help='batch size for testing: default 64')
    parser.add_argument('--batch-size-onnx', type=int, default=1, help='batch size for onnx: default 1')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--seed', type=int, default=12345, help='random seed: default 12345')
    parser.add_argument('--skip-eval-accuracy', action='store_true', help='Skip the accuracy evaluation after the QDQ insert/Calibration/QAT-Fintuning')

    # setting for calibration
    parser.add_argument('--hyp', type=str, default='data/hyp.qat.yaml', help='hyperparameters path')
    parser.add_argument('--calib-batch-size', type=int,
                        default=32, help='calib batch size: default 64')
    parser.add_argument('--num-calib-batch', default=16, type=int,
                        help='Number of batches for calibration. 0 will disable calibration. (default: 4)')
    parser.add_argument('--num-finetune-epochs', default=15, type=int,
                        help='Number of epochs to fine tune. 0 will disable fine tune. (default: 0)')
    parser.add_argument('--calibrator', type=str, choices=["max", "histogram"], default="max")
    parser.add_argument('--percentile', nargs='+', type=float, default=[99.9, 99.99, 99.999, 99.9999])
    parser.add_argument('--sensitivity', action="store_true", help="Build sensitivity profile")
    parser.add_argument('--evaluate-onnx', action="store_true", help="Evaluate exported ONNX")
    parser.add_argument("--accu-tolerance", type=float, default=0.925, help="used by test, for coco 0.367+0.558")
    parser.add_argument('--skip-layers', action="store_true", help='Skip some sensitivity layers')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')

    return parser


def prepare_model(calibrator, hyp, opt, device):
    """
    Prepare the model for the quantization, including quant modules, settings and dataloaders.
    """
    # Use 'spawn' to avoid CUDA reinitialization with forked subprocess
    torch.multiprocessing.set_start_method('spawn')

    ## Initialize quantization, model and data loaders
    quant_desc_input = QuantDescriptor(calib_method=calibrator)
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    # Model
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    nc = 1 if opt.single_cls else int(data_dict['nc'])    # number of classes

    # # Dynamic module replacement using monkey patching.
    # # Monkey patching, take Conv2d for example, replace the Conv2d operator with quant_nn.QuantConv2d to enable FakeQuant
    # quant_modules.initialize()

    pretrained = opt.weights.endswith('.pt')
    if pretrained:
        ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    # # Disable the monkey patching.
    # quant_modules.deactivate()

    model.eval()
    model.cuda()

    train_path = data_dict['train']
    test_path = data_dict['val']

    gs = max(int(model.stride.max()), 32)                               # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]   # verify imgsz are gs-multiples

    # Train dataloader
    trainloader, dataset = create_dataloader(train_path, imgsz, opt.batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=-1,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Test dataloader
    testloader = create_dataloader(test_path, imgsz_test, opt.batch_size*2, gs, opt,  # testloader
                                   hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                   world_size=opt.world_size, workers=opt.workers,
                                   pad=0.5, prefix=colorstr('val: '))[0]

    # Calib dataloader
    calibloader = create_dataloader(train_path, imgsz, opt.calib_batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=-1,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))[0]

    return model, trainloader, testloader, calibloader, dataset


def evaluate_accuracy(model, opt, testloader):
    opt.task = 'val'
    results, _, _ = test.test(opt.data,
         weights=opt.weights,
         batch_size=opt.batch_size_test,
         model=model,
         dataloader=testloader,
         conf_thres=opt.conf_thres,
         iou_thres=opt.iou_thres,
         save_json=opt.save_json,
         opt=opt)

    map50 = list(results)[3]
    map = list(results)[2]
    return map50, map

def print_module_status(model):
    """
    print the setting of quant module for debugging
    """
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                print('With _calibrator: ', F'{name:40}: {module}', module._learn_amax)
            else:
                print('No _calibrator: ', F'{name:40}: {module}')


def main(cmdline_args):
    parser = get_parser()
    opt = parser.parse_args(cmdline_args)
    print(parser.description)
    print(opt)

    # Check the validity of parameters
    hyp, opt, device, tb_writer = check_and_set_params(opt)


    # Prepare the pretrained model and data loaders
    model, trainloader, testloader, calibloader, dataset = prepare_model(opt.calibrator,
                                                            hyp, opt, device)

    # Initial accuracy evaluation
    if not opt.skip_eval_accuracy:
        map50_initial, map_initial = evaluate_accuracy(model, opt, testloader)
        print('Initial evaluation: ',  "{:.3f}, {:.3f}".format(map50_initial, map_initial))

    # Calibrate the model
    with torch.no_grad():
        calibrate_model(
            model=model,
            model_name=opt.model_name,
            data_loader=calibloader,
            num_calib_batch=opt.num_calib_batch,
            calibrator=opt.calibrator,
            hist_percentile=opt.percentile,
            out_dir=opt.out_dir,
            device=device)

    # Evaluate after calibration
    if opt.num_calib_batch > 0 and (not opt.skip_eval_accuracy):
        map50_calibrated, map_calibrated = evaluate_accuracy(model, opt, testloader)
        print('Calibration evaluation:', "{:.3f}, {:.3f}".format(map50_calibrated, map_calibrated))
    else:
        map50_calibrated, map_calibrated = -1.0, -1.0

    # Build sensitivy profile
    if opt.sensitivity:
        build_sensitivity_profile(model, opt, testloader)

    # Skip the sensitive layer
    if opt.skip_layers:
        skip_sensitive_layers(model, opt, testloader)

    if opt.num_finetune_epochs > 0:
        # Finetune the model
        train(hyp, opt, device, tb_writer, model=model, dataloader=trainloader,
              testloader=testloader, dataset=dataset)

        # Evaluate after finetuning
        if not opt.skip_eval_accuracy:
            map50_finetuned, map_finetuned = evaluate_accuracy(model, opt, testloader)
            print('Finetune evaluation: ', "{:.3f}, {:.3f}".format(map50_finetuned, map_finetuned))
    else:
        map50_finetuned, map_finetuned = -1.0, -1.0

    # Export to ONNX
    onnx_filename = opt.ckpt_path.replace('.pt', '.onnx')
    export_onnx(model, onnx_filename, opt.batch_size_onnx, opt.dynamic)

    # Print summary
    if not opt.skip_eval_accuracy:
        print("Accuracy summary:")
        table = PrettyTable(['Stage','Top1'])
        table.align['Stage'] = "l"
        table.add_row( [ 'Initial',     "{:.3f}, {:.3f}".format(map50_initial, map_initial) ] )
        table.add_row( [ 'Calibrated',  "{:.3f}, {:.3f}".format(map50_calibrated, map_calibrated) ] )
        table.add_row( [ 'Finetuned',   "{:.3f}, {:.3f}".format(map50_finetuned, map_finetuned) ] )
        print(table)

    return 0


def export_onnx(model, onnx_filename, batch_onnx, dynamic_shape):
    model.model[-1].export = True       # Do not export Detect() layer grid
    model.eval()

    # We have to shift to pytorch's fake quant ops before exporting the model to ONNX
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    # Export ONNX for multiple batch sizes
    print("Creating ONNX file: " + onnx_filename)
    dummy_input = torch.randn(batch_onnx, 3, 640, 640, device='cuda')     #TODO: switch input dims by model

    try:
        import onnx
        torch.onnx.export(model, dummy_input, onnx_filename, verbose=False, opset_version=13, input_names=['images'],
                          output_names= ['output_0', 'output_1', 'output_2'],
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'}} if dynamic_shape else None,
                          do_constant_folding=True)

        # Checks enable_onnx_checker=False, 
        onnx_model = onnx.load(onnx_filename)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print('ONNX export success, saved as %s' % onnx_filename)
    except ValueError:
        warnings.warn(
            UserWarning("Per-channel quantization is not yet supported in Pytorch/ONNX RT (requires ONNX opset 13)"))
        print("Failed to export to ONNX")
        return False

    # Restore the PSX/TensorRT's fake quant mechanism
    quant_nn.TensorQuantizer.use_fb_fake_quant = False
    # Restore the model to train/test mode, use Detect() layer grid
    model.model[-1].export = False

    return True


def calibrate_model(model, model_name, data_loader, num_calib_batch, calibrator, hist_percentile, out_dir, device):
    """
        Feed data to the network and calibrate.
        Arguments:
            model: classification model
            model_name: name to use when creating state files
            data_loader: calibration data set
            num_calib_batch: amount of calibration passes to perform
            calibrator: type of calibration to use (max/histogram)
            hist_percentile: percentiles to be used for historgram calibration
            out_dir: dir to save state files in
    """

    if num_calib_batch > 0:
        print("Calibrating model")
        with torch.no_grad():
            collect_stats(model, data_loader, num_calib_batch, device)

        if not calibrator == "histogram":
            compute_amax(model, method="max")
            calib_output = os.path.join(
                out_dir,
                F"{model_name}-max-{num_calib_batch*data_loader.batch_size}.pth")

            ckpt = {'model': deepcopy(model)}
            torch.save(ckpt, calib_output)
        else:
            for percentile in hist_percentile:
                print(F"{percentile} percentile calibration")
                compute_amax(model, method="percentile")
                calib_output = os.path.join(
                    out_dir,
                    F"{model_name}-percentile-{percentile}-{num_calib_batch*data_loader.batch_size}.pth")

                ckpt = {'model': deepcopy(model)}
                torch.save(ckpt, calib_output)

            for method in ["mse", "entropy"]:
                print(F"{method} calibration")
                compute_amax(model, method=method)

                calib_output = os.path.join(
                    out_dir,
                    F"{model_name}-{method}-{num_calib_batch*data_loader.batch_size}.pth")

                ckpt = {'model': deepcopy(model)}
                torch.save(ckpt, calib_output)

def collect_stats(model, data_loader, num_batches, device):
    """Feed data to the network and collect statistics"""
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, (img, _, _, _) in tqdm(enumerate(data_loader), total=num_batches):
        img = img.to(device, non_blocking=True)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        model(img)
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            # print(F"{name:40}: {module}")
    model.cuda()


def build_sensitivity_profile(model, opt, testloader):
    quant_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.disable()
            layer_name = name.replace("._input_quantizer", "").replace("._weight_quantizer", "")
            if layer_name not in quant_layer_names:
                quant_layer_names.append(layer_name)
    print(F"{len(quant_layer_names)} quantized layers found.")

    # Build sensitivity profile
    quant_layer_sensitivity = {}
    for i, quant_layer in enumerate(quant_layer_names):
        print(F"Enable {quant_layer}")
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer) and quant_layer in name:
                module.enable()
                print(F"{name:40}: {module}")

        # Eval the model
        map50, map50_95 = evaluate_accuracy(model, opt, testloader)
        print(F"mAP@IoU=0.50: {map50}, mAP@IoU=0.50:0.95: {map50_95}")
        quant_layer_sensitivity[quant_layer] = opt.accu_tolerance - (map50 + map50_95)

        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer) and quant_layer in name:
                module.disable()
                print(F"{name:40}: {module}")

    # Skip most sensitive layers until accuracy target is met
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.enable()
    quant_layer_sensitivity = collections.OrderedDict(sorted(quant_layer_sensitivity.items(), key=lambda x: x[1]))
    print(quant_layer_sensitivity)

    skipped_layers = []
    for quant_layer, _ in quant_layer_sensitivity.items():
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if quant_layer in name:
                    print(F"Disable {name}")
                    if not quant_layer in skipped_layers:
                        skipped_layers.append(quant_layer)
                    module.disable()
        map50, map50_95 = evaluate_accuracy(model, opt, testloader)
        if (map50 + map50_95) >= opt.accu_tolerance - 0.05:
            print(F"Accuracy tolerance {opt.accu_tolerance} is met by skipping {len(skipped_layers)} sensitive layers.")
            print(skipped_layers)
            onnx_filename = opt.ckpt_path.replace('.pt', F'_skip{len(skipped_layers)}.onnx')
            export_onnx(model, onnx_filename, opt.batch_size_onnx, opt.dynamic)
            return
    raise ValueError(f"Accuracy tolerance {opt.accu_tolerance} can not be met with any layer quantized!")


def skip_sensitive_layers(model, opt, testloader):
    print('Skip the sensitive layers.')
    # Sensitivity layers for yolov5s
    skipped_layers = ['model.1.conv',          # the first conv
                      'model.2.cv1.conv',      # the second conv
                      'model.24.m.2',          # detect layer
                      'model.24.m.1',          # detect layer
                      ]

    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            layer_name = name.replace("._input_quantizer", "").replace("._weight_quantizer", "")
            if layer_name in skipped_layers:
                print(F"Disable {name}")
                module.disable()

    map50, map50_95 = evaluate_accuracy(model, opt, testloader)
    print(F"mAP@IoU=0.50: {map50}, mAP@IoU=0.50:0.95: {map50_95}")

    onnx_filename = opt.ckpt_path.replace('.pt', F'_skip{len(skipped_layers)}.onnx')
    export_onnx(model, onnx_filename, opt.batch_size_onnx, opt.dynamic)
    return


if __name__ == '__main__':
    res = main(sys.argv[1:])
    exit(res)
