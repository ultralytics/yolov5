import argparse
from copy import deepcopy
from pathlib import Path
import sys, traceback
import os
import logging

import torch
import torch.nn as nn
import tensorflow as tf
if tf.__version__.startswith('1'):
    tf.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from models.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, Concat, autopad
from models.experimental import MixConv2d, CrossConv, C3
from utils.general import check_anchor_order, make_divisible, check_file, set_logging
from utils.torch_utils import (
    time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, select_device)
from models.yolo import Detect


logger = logging.getLogger(__name__)


class tf_BN(keras.layers.Layer):
    # TensorFlow BatchNormalization wrapper
    def __init__(self, w=None):
        super(tf_BN, self).__init__()
        self.bn = keras.layers.BatchNormalization(
                beta_initializer=keras.initializers.Constant(w.bias.numpy()),
                gamma_initializer=keras.initializers.Constant(w.weight.numpy()),
                moving_mean_initializer=keras.initializers.Constant(w.running_mean.numpy()),
                moving_variance_initializer=keras.initializers.Constant(w.running_var.numpy()))

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
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):  # ch_in, ch_out, weights, kernel, stride, padding, groups
        super(tf_Conv, self).__init__()
        assert g == 1, "TF v2.2 Conv2D does not support 'groups' argument"
        assert isinstance(k, int), "Convolution with multiple kernels are not allowed."

        # TensorFlow convolution padding is inconsistent with PyTorch (e.g. k=3 s=2 'SAME' padding)
        # see https://stackoverflow.com/questions/52975843/comparing-conv2d-with-padding-between-tensorflow-and-pytorch
        if s == 1:
            self.conv = keras.layers.Conv2D(
                    c2, k, s, 'SAME', use_bias=False,
                    kernel_initializer=
                            keras.initializers.Constant(w.conv.weight.permute(2, 3, 1, 0).numpy()))
        else:
            self.pad = tf_Pad(autopad(k, p))
            self.conv = keras.Sequential([
                self.pad, 
                keras.layers.Conv2D(
                    c2, k, s, 'VALID', use_bias=False,
                    kernel_initializer=
                            keras.initializers.Constant(w.conv.weight.permute(2, 3, 1, 0).numpy()))
            ])

        self.bn = tf_BN(w.bn)

        # YOLOv5 v3 uses Hardswish for activations
        if isinstance(w.act, nn.LeakyReLU):
            self.act = (lambda x: keras.activations.relu(x, alpha=0.1)) if act else tf.identity
        elif isinstance(w.act, nn.Hardswish): 
            self.act = (lambda x: x * tf.nn.relu6(x+3) * 0.166666667) if act else tf.identity

    def call(self, inputs):
        return self.act(self.bn(self.conv(inputs)))


class tf_Focus(keras.layers.Layer):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):  # ch_in, ch_out, kernel, stride, padding, groups
        super(tf_Focus, self).__init__()
        self.conv = tf_Conv(c1 * 4, c2, k, s, p, g, act, w.conv)

    def call(self, inputs):  # x(b,w,h,c) -> y(b,w/2,h/2,4c)
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
                kernel_initializer=
                        keras.initializers.Constant(w.weight.permute(2, 3, 1, 0).numpy()),
                bias_initializer=
                        keras.initializers.Constant(w.bias.numpy()) if bias else None,
                )

    def call(self, inputs):
        return self.conv(inputs)


class tf_BottleneckCSP(keras.layers.Layer):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):  # ch_in, ch_out, number, shortcut, groups, expansion
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
        # self.stride = None  # strides computed during build
        self.stride = tf.Variable([8, 16, 32], dtype=tf.float32)
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [tf.zeros(1)] * self.nl  # init grid
        a = tf.reshape(tf.convert_to_tensor(anchors, dtype=tf.float32), [self.nl, -1 ,2])
        self.anchors = tf.Variable(a, trainable=False)
        self.anchor_grid = tf.reshape(tf.Variable(a, trainable=False), [self.nl, 1, -1, 1, 2])
        self.m = [tf_Conv2d(x, self.no * self.na, 1, w=w.m[i]) for i, x in enumerate(ch)]
        self.export = False  # onnx export
        self.training = True # set to False after building model
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
                squared = 4 * y[..., 2:4] * y[..., 2:4]
                wh = squared * self.anchor_grid[i]  # wh
                y = tf.concat([xy, wh, y[..., 4:]], -1)
                z.append(tf.reshape(y, [opt.batch_size, 3 * ny * nx, self.no]))

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


def tf_check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = tf.reshape(tf.math.reduce_prod(m.anchor_grid, -1), [-1])
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    # Add numpy() in comparison for TensorFlow v1.15
    if tf.math.sign(da).numpy() != tf.math.sign(ds).numpy():  # same order
        print('Reversing anchor order')
        m.anchors[:] = tf.reverse(m.anchors, 0)
        m.anchor_grid[:] = tf.reverse(m.anchor_grid, [0])


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
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model.layers[-1]  # Detect()
        if isinstance(m, tf_Detect):
            # s = 128  # 2x min stride
            # m.stride = tf.convert_to_tensor([s / x.shape[-2] for x in self.predict(tf.zeros([1, s, s, ch]))])  # forward
            m.anchors = tf.cast(m.anchors, dtype=tf.float32) / tf.reshape(m.stride, [-1, 1, 1])
            tf_check_anchor_order(m)
            # self.stride = m.stride
            # self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        # torch_utils.initialize_weights(self)

        # keras.layers.Layer has no summary/info method
        # self.info()

    def predict(self, inputs, profile=False):
        y, dt = [], []  # outputs
        x = inputs
        for i, m in enumerate(self.model.layers):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                try:
                    import thop
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
                except:
                    o = 0
                t = torch_utils.time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((torch_utils.time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.savelist else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./models/yolov5s.yaml', help='cfg path')
    parser.add_argument('--weights', type=str, default='./weights/yolov5s.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)

    # Input
    img = torch.zeros((opt.batch_size, 3, *opt.img_size))  # image size(1,3,320,192) iDetection

    # Load PyTorch model
    model = torch.load(opt.weights, map_location=torch.device('cpu'))['model'].float()
    model.eval()
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

        inputs = keras.Input(shape=(*opt.img_size, 3))
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
    try:
        print('\nStarting TFLite export with TensorFlow %s...' % tf.__version__)
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.allow_custom_ops = False
        converter.experimental_new_converter = True
        tflite_model = converter.convert()
        f = opt.weights.replace('.pt', '.tflite')  # filename
        open(f, "wb").write(tflite_model)
        print('TFLite export success, saved as %s' % f)
    except Exception as e:
        print('TFLite export failure: %s' % e)
        traceback.print_exc(file=sys.stdout)

