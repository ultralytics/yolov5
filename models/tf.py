import argparse
import logging
import os
import sys
import traceback
from copy import deepcopy
from pathlib import Path

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import yaml
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from models.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, Concat, autopad, C3
from models.experimental import MixConv2d, CrossConv, attempt_load
from models.yolo import Detect
from utils.datasets import LoadImages
from utils.general import make_divisible, check_file, check_dataset
from utils.google_utils import attempt_download

logger = logging.getLogger(__name__)


class tf_BN(keras.layers.Layer):
    # TensorFlow BatchNormalization wrapper
    def __init__(self, w=None):
        super(tf_BN, self).__init__()
        self.bn = keras.layers.BatchNormalization(
            beta_initializer=keras.initializers.Constant(w.bias.numpy()),
            gamma_initializer=keras.initializers.Constant(w.weight.numpy()),
            moving_mean_initializer=keras.initializers.Constant(w.running_mean.numpy()),
            moving_variance_initializer=keras.initializers.Constant(w.running_var.numpy()),
            epsilon=w.eps)

    def call(self, inputs):
        return self.bn(inputs)


class tf_Pad(keras.layers.Layer):
    def __init__(self, pad):
        super(tf_Pad, self).__init__()
        self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])

    def call(self, inputs):
        return tf.pad(inputs, self.pad, mode='constant', constant_values=0)


class tf_Conv(keras.layers.Layer):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):
        # ch_in, ch_out, weights, kernel, stride, padding, groups
        super(tf_Conv, self).__init__()
        assert g == 1, "TF v2.2 Conv2D does not support 'groups' argument"
        assert isinstance(k, int), "Convolution with multiple kernels are not allowed."
        # TensorFlow convolution padding is inconsistent with PyTorch (e.g. k=3 s=2 'SAME' padding)
        # see https://stackoverflow.com/questions/52975843/comparing-conv2d-with-padding-between-tensorflow-and-pytorch

        conv = keras.layers.Conv2D(
            c2, k, s, 'SAME' if s == 1 else 'VALID', use_bias=False,
            kernel_initializer=keras.initializers.Constant(w.conv.weight.permute(2, 3, 1, 0).numpy()))
        self.conv = conv if s == 1 else keras.Sequential([tf_Pad(autopad(k, p)), conv])
        self.bn = tf_BN(w.bn) if hasattr(w, 'bn') else tf.identity

        # YOLOv5 activations
        if isinstance(w.act, nn.LeakyReLU):
            self.act = (lambda x: keras.activations.relu(x, alpha=0.1)) if act else tf.identity
        elif isinstance(w.act, nn.Hardswish):
            self.act = (lambda x: x * tf.nn.relu6(x + 3) * 0.166666667) if act else tf.identity
        elif isinstance(w.act, nn.SiLU):
            self.act = (lambda x: keras.activations.swish(x)) if act else tf.identity

    def call(self, inputs):
        return self.act(self.bn(self.conv(inputs)))


class tf_Focus(keras.layers.Layer):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):
        # ch_in, ch_out, kernel, stride, padding, groups
        super(tf_Focus, self).__init__()
        self.conv = tf_Conv(c1 * 4, c2, k, s, p, g, act, w.conv)

    def call(self, inputs):  # x(b,w,h,c) -> y(b,w/2,h/2,4c)
        # inputs = inputs / 255.  # normalize 0-255 to 0-1
        return self.conv(tf.concat([inputs[:, ::2, ::2, :],
                                    inputs[:, 1::2, ::2, :],
                                    inputs[:, ::2, 1::2, :],
                                    inputs[:, 1::2, 1::2, :]], 3))


class tf_Bottleneck(keras.layers.Layer):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, w=None):  # ch_in, ch_out, shortcut, groups, expansion
        super(tf_Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = tf_Conv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = tf_Conv(c_, c2, 3, 1, g=g, w=w.cv2)
        self.add = shortcut and c1 == c2

    def call(self, inputs):
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs))


class tf_Conv2d(keras.layers.Layer):
    # Substitution for PyTorch nn.Conv2D
    def __init__(self, c1, c2, k, s=1, g=1, bias=True, w=None):
        super(tf_Conv2d, self).__init__()
        assert g == 1, "TF v2.2 Conv2D does not support 'groups' argument"
        self.conv = keras.layers.Conv2D(
            c2, k, s, 'VALID', use_bias=bias,
            kernel_initializer=keras.initializers.Constant(w.weight.permute(2, 3, 1, 0).numpy()),
            bias_initializer=keras.initializers.Constant(w.bias.numpy()) if bias else None, )

    def call(self, inputs):
        return self.conv(inputs)


class tf_BottleneckCSP(keras.layers.Layer):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super(tf_BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = tf_Conv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = tf_Conv2d(c1, c_, 1, 1, bias=False, w=w.cv2)
        self.cv3 = tf_Conv2d(c_, c_, 1, 1, bias=False, w=w.cv3)
        self.cv4 = tf_Conv(2 * c_, c2, 1, 1, w=w.cv4)
        self.bn = tf_BN(w.bn)
        self.act = lambda x: keras.activations.relu(x, alpha=0.1)
        self.m = keras.Sequential([tf_Bottleneck(c_, c_, shortcut, g, e=1.0, w=w.m[j]) for j in range(n)])

    def call(self, inputs):
        y1 = self.cv3(self.m(self.cv1(inputs)))
        y2 = self.cv2(inputs)
        return self.cv4(self.act(self.bn(tf.concat((y1, y2), axis=3))))


class tf_C3(keras.layers.Layer):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super(tf_C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = tf_Conv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = tf_Conv(c1, c_, 1, 1, w=w.cv2)
        self.cv3 = tf_Conv(2 * c_, c2, 1, 1, w=w.cv3)
        self.m = keras.Sequential([tf_Bottleneck(c_, c_, shortcut, g, e=1.0, w=w.m[j]) for j in range(n)])

    def call(self, inputs):
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3))


class tf_SPP(keras.layers.Layer):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), w=None):
        super(tf_SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = tf_Conv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = tf_Conv(c_ * (len(k) + 1), c2, 1, 1, w=w.cv2)
        self.m = [keras.layers.MaxPool2D(pool_size=x, strides=1, padding='SAME') for x in k]

    def call(self, inputs):
        x = self.cv1(inputs)
        return self.cv2(tf.concat([x] + [m(x) for m in self.m], 3))


class tf_Detect(keras.layers.Layer):
    def __init__(self, nc=80, anchors=(), ch=(), w=None):  # detection layer
        super(tf_Detect, self).__init__()
        self.stride = tf.convert_to_tensor(w.stride.numpy(), dtype=tf.float32)
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [tf.zeros(1)] * self.nl  # init grid
        self.anchors = tf.convert_to_tensor(w.anchors.numpy(), dtype=tf.float32)
        self.anchor_grid = tf.reshape(tf.convert_to_tensor(w.anchor_grid.numpy(), dtype=tf.float32),
                                      [self.nl, 1, -1, 1, 2])
        self.m = [tf_Conv2d(x, self.no * self.na, 1, w=w.m[i]) for i, x in enumerate(ch)]
        self.export = False  # onnx export
        self.training = True  # set to False after building model
        for i in range(self.nl):
            ny, nx = opt.img_size[0] // self.stride[i], opt.img_size[1] // self.stride[i]
            self.grid[i] = self._make_grid(nx, ny)

    def call(self, inputs):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        x = []
        for i in range(self.nl):
            x.append(self.m[i](inputs[i]))
            # x(bs,20,20,255) to x(bs,3,20,20,85)
            ny, nx = opt.img_size[0] // self.stride[i], opt.img_size[1] // self.stride[i]
            x[i] = tf.transpose(tf.reshape(x[i], [-1, ny * nx, self.na, self.no]), [0, 2, 1, 3])

            if not self.training:  # inference
                y = tf.sigmoid(x[i])
                xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                # Normalize xywh to 0-1 to reduce calibration error
                xy /= tf.constant([[opt.img_size[1], opt.img_size[0]]], dtype=tf.float32)
                wh /= tf.constant([[opt.img_size[1], opt.img_size[0]]], dtype=tf.float32)
                y = tf.concat([xy, wh, y[..., 4:]], -1)
                z.append(tf.reshape(y, [-1, 3 * ny * nx, self.no]))

        return x if self.training else (tf.concat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        # yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        # return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
        xv, yv = tf.meshgrid(tf.range(nx), tf.range(ny))
        return tf.cast(tf.reshape(tf.stack([xv, yv], 2), [1, 1, ny * nx, 2]), dtype=tf.float32)


class tf_Upsample(keras.layers.Layer):
    def __init__(self, size, scale_factor, mode, w=None):
        super(tf_Upsample, self).__init__()
        assert scale_factor == 2, "scale_factor must be 2"
        # self.upsample = keras.layers.UpSampling2D(size=scale_factor, interpolation=mode)
        if opt.tf_raw_resize:
            # with default arguments: align_corners=False, half_pixel_centers=False
            self.upsample = lambda x: tf.raw_ops.ResizeNearestNeighbor(images=x,
                                                                       size=(x.shape[1] * 2, x.shape[2] * 2))
        else:
            self.upsample = lambda x: tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method=mode)

    def call(self, inputs):
        return self.upsample(inputs)


class tf_Concat(keras.layers.Layer):
    def __init__(self, dimension=1, w=None):
        super(tf_Concat, self).__init__()
        assert dimension == 1, "convert only NCHW to NHWC concat"
        self.d = 3

    def call(self, inputs):
        return tf.concat(inputs, self.d)


def parse_model(d, ch, model):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m_str = m
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        tf_m = eval('tf_' + m_str.replace('nn.', ''))
        m_ = keras.Sequential([tf_m(*args, w=model.model[i][j]) for j in range(n)]) if n > 1 \
            else tf_m(*args, w=model.model[i])  # module

        torch_m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in torch_m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return keras.Sequential(layers), sorted(save)


class tf_Model():
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, model=None):  # model, input channels, number of classes
        super(tf_Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc and nc != self.yaml['nc']:
            print('Overriding %s nc=%g with nc=%g' % (cfg, self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.savelist = parse_model(deepcopy(self.yaml), ch=[ch], model=model)  # model, savelist, ch_out

    def predict(self, inputs, profile=False):
        y = []  # outputs
        x = inputs
        for i, m in enumerate(self.model.layers):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.savelist else None)  # save output

        # Add TensorFlow NMS
        if opt.tf_nms:
            boxes = tf.expand_dims(xywh2xyxy(x[0][..., :4]), 2)
            probs = x[0][:, :, 4:5]
            classes = x[0][:, :, 5:]
            scores = probs * classes
            nms = tf.image.combined_non_max_suppression(
                boxes, scores, opt.topk_per_class, opt.topk_all, opt.iou_thres, opt.score_thres, clip_boxes=False)
            return nms, x[1]

        return x[0]  # output only first tensor [1,6300,85] = [xywh, conf, class0, class1, ...]
        # x = x[0][0]  # [x(1,6300,85), ...] to x(6300,85)
        # xywh = x[..., :4]  # x(6300,4) boxes
        # conf = x[..., 4:5]  # x(6300,1) confidences
        # cls = tf.reshape(tf.cast(tf.argmax(x[..., 5:], axis=1), tf.float32), (-1, 1))  # x(6300,1)  classes
        # return tf.concat([conf, cls, xywh], 1)


def xywh2xyxy(xywh):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = tf.split(xywh, num_or_size_splits=4, axis=-1)
    return tf.concat([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)


def representative_dataset_gen():
    # Representative dataset for use with converter.representative_dataset
    n = 0
    for path, img, im0s, vid_cap in dataset:
        # Get sample input data as a numpy array in a method of your choosing.
        n += 1
        input = np.transpose(img, [1, 2, 0])
        input = np.expand_dims(input, axis=0).astype(np.float32)
        input /= 255.0
        yield [input]
        if n >= opt.ncalib:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='cfg path')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[320, 320], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic-batch-size', action='store_true', help='dynamic batch size')
    parser.add_argument('--source', type=str, default='../data/coco128.yaml', help='dir of images or data.yaml file')
    parser.add_argument('--ncalib', type=int, default=100, help='number of calibration images')
    parser.add_argument('--tfl-int8', action='store_true', dest='tfl_int8', help='export TFLite int8 model')
    parser.add_argument('--tf-nms', action='store_true', dest='tf_nms', help='TF NMS (without TFLite export)')
    parser.add_argument('--tf-raw-resize', action='store_true', dest='tf_raw_resize',
                        help='use tf.raw_ops.ResizeNearestNeighbor for resize')
    parser.add_argument('--topk-per-class', type=int, default=100, help='topk per class to keep in NMS')
    parser.add_argument('--topk-all', type=int, default=100, help='topk for all classes to keep in NMS')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--score-thres', type=float, default=0.4, help='score threshold for NMS')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)

    # Input
    img = torch.zeros((opt.batch_size, 3, *opt.img_size))  # image size(1,3,320,192) iDetection

    # Load PyTorch model
    model = attempt_load(opt.weights, map_location=torch.device('cpu'), inplace=True, fuse=False)
    model.model[-1].export = False  # set Detect() layer export=True
    y = model(img)  # dry run
    nc = y[0].shape[-1] - 5

    # TensorFlow saved_model export
    try:
        print('\nStarting TensorFlow saved_model export with TensorFlow %s...' % tf.__version__)
        tf_model = tf_Model(opt.cfg, model=model, nc=nc)
        img = tf.zeros((opt.batch_size, *opt.img_size, 3))  # NHWC Input for TensorFlow

        m = tf_model.model.layers[-1]
        assert isinstance(m, tf_Detect), "the last layer must be Detect"
        m.training = False
        y = tf_model.predict(img)

        inputs = keras.Input(shape=(*opt.img_size, 3), batch_size=None if opt.dynamic_batch_size else opt.batch_size)
        keras_model = keras.Model(inputs=inputs, outputs=tf_model.predict(inputs))
        keras_model.summary()
        path = opt.weights.replace('.pt', '_saved_model')  # filename
        keras_model.save(path, save_format='tf')
        print('TensorFlow saved_model export success, saved as %s' % path)
    except Exception as e:
        print('TensorFlow saved_model export failure: %s' % e)
        traceback.print_exc(file=sys.stdout)

    # TensorFlow GraphDef export
    try:
        print('\nStarting TensorFlow GraphDef export with TensorFlow %s...' % tf.__version__)

        # https://github.com/leimao/Frozen_Graph_TensorFlow
        full_model = tf.function(lambda x: keras_model(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))

        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        f = opt.weights.replace('.pt', '.pb')  # filename
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir=os.path.dirname(f),
                          name=os.path.basename(f),
                          as_text=False)

        print('TensorFlow GraphDef export success, saved as %s' % f)
    except Exception as e:
        print('TensorFlow GraphDef export failure: %s' % e)
        traceback.print_exc(file=sys.stdout)

    # TFLite model export
    if not opt.tf_nms:
        try:
            print('\nStarting TFLite export with TensorFlow %s...' % tf.__version__)

            # fp32 TFLite model export ---------------------------------------------------------------------------------
            # converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
            # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            # converter.allow_custom_ops = False
            # converter.experimental_new_converter = True
            # tflite_model = converter.convert()
            # f = opt.weights.replace('.pt', '.tflite')  # filename
            # open(f, "wb").write(tflite_model)

            # fp16 TFLite model export ---------------------------------------------------------------------------------
            converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # converter.representative_dataset = representative_dataset_gen
            # converter.target_spec.supported_types = [tf.float16]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            converter.allow_custom_ops = False
            converter.experimental_new_converter = True
            tflite_model = converter.convert()
            f = opt.weights.replace('.pt', '-fp16.tflite')  # filename
            open(f, "wb").write(tflite_model)
            print('\nTFLite export success, saved as %s' % f)

            # int8 TFLite model export ---------------------------------------------------------------------------------
            if opt.tfl_int8:
                # Representative Dataset
                if opt.source.endswith('.yaml'):
                    with open(check_file(opt.source)) as f:
                        data = yaml.load(f, Loader=yaml.FullLoader)  # data dict
                        check_dataset(data)  # check
                    opt.source = data['train']
                dataset = LoadImages(opt.source, img_size=opt.img_size, auto=False)
                converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = representative_dataset_gen
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8  # or tf.int8
                converter.inference_output_type = tf.uint8  # or tf.int8
                converter.allow_custom_ops = False
                converter.experimental_new_converter = True
                converter.experimental_new_quantizer = False
                tflite_model = converter.convert()
                f = opt.weights.replace('.pt', '-int8.tflite')  # filename
                open(f, "wb").write(tflite_model)
                print('\nTFLite (int8) export success, saved as %s' % f)

        except Exception as e:
            print('\nTFLite export failure: %s' % e)
            traceback.print_exc(file=sys.stdout)
