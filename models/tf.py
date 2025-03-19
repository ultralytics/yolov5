# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
TensorFlow, Keras and TFLite versions of YOLOv5
Authored by https://github.com/zldrobit in PR https://github.com/ultralytics/yolov5/pull/1127.

Usage:
    $ python models/tf.py --weights yolov5s.pt

Export:
    $ python export.py --weights yolov5s.pt --include saved_model pb tflite tfjs
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from tensorflow import keras

from models.common import (
    C3,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3x,
    Concat,
    Conv,
    CrossConv,
    DWConv,
    DWConvTranspose2d,
    Focus,
    autopad,
)
from models.experimental import MixConv2d, attempt_load
from models.yolo import Detect, Segment
from utils.activations import SiLU
from utils.general import LOGGER, make_divisible, print_args


class TFBN(keras.layers.Layer):
    """TensorFlow BatchNormalization wrapper for initializing with optional pretrained weights."""

    def __init__(self, w=None):
        """Initializes a TensorFlow BatchNormalization layer with optional pretrained weights."""
        super().__init__()
        self.bn = keras.layers.BatchNormalization(
            beta_initializer=keras.initializers.Constant(w.bias.numpy()),
            gamma_initializer=keras.initializers.Constant(w.weight.numpy()),
            moving_mean_initializer=keras.initializers.Constant(w.running_mean.numpy()),
            moving_variance_initializer=keras.initializers.Constant(w.running_var.numpy()),
            epsilon=w.eps,
        )

    def call(self, inputs):
        """Applies batch normalization to the inputs."""
        return self.bn(inputs)


class TFPad(keras.layers.Layer):
    """Pads input tensors in spatial dimensions 1 and 2 with specified integer or tuple padding values."""

    def __init__(self, pad):
        """
        Initializes a padding layer for spatial dimensions 1 and 2 with specified padding, supporting both int and tuple
        inputs.

        Inputs are
        """
        super().__init__()
        if isinstance(pad, int):
            self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
        else:  # tuple/list
            self.pad = tf.constant([[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]])

    def call(self, inputs):
        """Pads input tensor with zeros using specified padding, suitable for int and tuple pad dimensions."""
        return tf.pad(inputs, self.pad, mode="constant", constant_values=0)


class TFConv(keras.layers.Layer):
    """Implements a standard convolutional layer with optional batch normalization and activation for TensorFlow."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):
        """
        Initializes a standard convolution layer with optional batch normalization and activation; supports only
        group=1.

        Inputs are ch_in, ch_out, weights, kernel, stride, padding, groups.
        """
        super().__init__()
        assert g == 1, "TF v2.2 Conv2D does not support 'groups' argument"
        # TensorFlow convolution padding is inconsistent with PyTorch (e.g. k=3 s=2 'SAME' padding)
        # see https://stackoverflow.com/questions/52975843/comparing-conv2d-with-padding-between-tensorflow-and-pytorch
        conv = keras.layers.Conv2D(
            filters=c2,
            kernel_size=k,
            strides=s,
            padding="SAME" if s == 1 else "VALID",
            use_bias=not hasattr(w, "bn"),
            kernel_initializer=keras.initializers.Constant(w.conv.weight.permute(2, 3, 1, 0).numpy()),
            bias_initializer="zeros" if hasattr(w, "bn") else keras.initializers.Constant(w.conv.bias.numpy()),
        )
        self.conv = conv if s == 1 else keras.Sequential([TFPad(autopad(k, p)), conv])
        self.bn = TFBN(w.bn) if hasattr(w, "bn") else tf.identity
        self.act = activations(w.act) if act else tf.identity

    def call(self, inputs):
        """Applies convolution, batch normalization, and activation function to input tensors."""
        return self.act(self.bn(self.conv(inputs)))


class TFDWConv(keras.layers.Layer):
    """Initializes a depthwise convolution layer with optional batch normalization and activation for TensorFlow."""

    def __init__(self, c1, c2, k=1, s=1, p=None, act=True, w=None):
        """
        Initializes a depthwise convolution layer with optional batch normalization and activation for TensorFlow
        models.

        Input are ch_in, ch_out, weights, kernel, stride, padding, groups.
        """
        super().__init__()
        assert c2 % c1 == 0, f"TFDWConv() output={c2} must be a multiple of input={c1} channels"
        conv = keras.layers.DepthwiseConv2D(
            kernel_size=k,
            depth_multiplier=c2 // c1,
            strides=s,
            padding="SAME" if s == 1 else "VALID",
            use_bias=not hasattr(w, "bn"),
            depthwise_initializer=keras.initializers.Constant(w.conv.weight.permute(2, 3, 1, 0).numpy()),
            bias_initializer="zeros" if hasattr(w, "bn") else keras.initializers.Constant(w.conv.bias.numpy()),
        )
        self.conv = conv if s == 1 else keras.Sequential([TFPad(autopad(k, p)), conv])
        self.bn = TFBN(w.bn) if hasattr(w, "bn") else tf.identity
        self.act = activations(w.act) if act else tf.identity

    def call(self, inputs):
        """Applies convolution, batch normalization, and activation function to input tensors."""
        return self.act(self.bn(self.conv(inputs)))


class TFDWConvTranspose2d(keras.layers.Layer):
    """Implements a depthwise ConvTranspose2D layer for TensorFlow with specific settings."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0, w=None):
        """
        Initializes depthwise ConvTranspose2D layer with specific channel, kernel, stride, and padding settings.

        Inputs are ch_in, ch_out, weights, kernel, stride, padding, groups.
        """
        super().__init__()
        assert c1 == c2, f"TFDWConv() output={c2} must be equal to input={c1} channels"
        assert k == 4 and p1 == 1, "TFDWConv() only valid for k=4 and p1=1"
        weight, bias = w.weight.permute(2, 3, 1, 0).numpy(), w.bias.numpy()
        self.c1 = c1
        self.conv = [
            keras.layers.Conv2DTranspose(
                filters=1,
                kernel_size=k,
                strides=s,
                padding="VALID",
                output_padding=p2,
                use_bias=True,
                kernel_initializer=keras.initializers.Constant(weight[..., i : i + 1]),
                bias_initializer=keras.initializers.Constant(bias[i]),
            )
            for i in range(c1)
        ]

    def call(self, inputs):
        """Processes input through parallel convolutions and concatenates results, trimming border pixels."""
        return tf.concat([m(x) for m, x in zip(self.conv, tf.split(inputs, self.c1, 3))], 3)[:, 1:-1, 1:-1]


class TFFocus(keras.layers.Layer):
    """Focuses spatial information into channel space using pixel shuffling and convolution for TensorFlow models."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):
        """
        Initializes TFFocus layer to focus width and height information into channel space with custom convolution
        parameters.

        Inputs are ch_in, ch_out, kernel, stride, padding, groups.
        """
        super().__init__()
        self.conv = TFConv(c1 * 4, c2, k, s, p, g, act, w.conv)

    def call(self, inputs):
        """
        Performs pixel shuffling and convolution on input tensor, downsampling by 2 and expanding channels by 4.

        Example x(b,w,h,c) -> y(b,w/2,h/2,4c).
        """
        inputs = [inputs[:, ::2, ::2, :], inputs[:, 1::2, ::2, :], inputs[:, ::2, 1::2, :], inputs[:, 1::2, 1::2, :]]
        return self.conv(tf.concat(inputs, 3))


class TFBottleneck(keras.layers.Layer):
    """Implements a TensorFlow bottleneck layer with optional shortcut connections for efficient feature extraction."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, w=None):
        """
        Initializes a standard bottleneck layer for TensorFlow models, expanding and contracting channels with optional
        shortcut.

        Arguments are ch_in, ch_out, shortcut, groups, expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_, c2, 3, 1, g=g, w=w.cv2)
        self.add = shortcut and c1 == c2

    def call(self, inputs):
        """Performs forward pass; if shortcut is True & input/output channels match, adds input to the convolution
        result.
        """
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs))


class TFCrossConv(keras.layers.Layer):
    """Implements a cross convolutional layer with optional expansion, grouping, and shortcut for TensorFlow."""

    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False, w=None):
        """Initializes cross convolution layer with optional expansion, grouping, and shortcut addition capabilities."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, (1, k), (1, s), w=w.cv1)
        self.cv2 = TFConv(c_, c2, (k, 1), (s, 1), g=g, w=w.cv2)
        self.add = shortcut and c1 == c2

    def call(self, inputs):
        """Passes input through two convolutions optionally adding the input if channel dimensions match."""
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs))


class TFConv2d(keras.layers.Layer):
    """Implements a TensorFlow 2D convolution layer, mimicking PyTorch's nn.Conv2D for specified filters and stride."""

    def __init__(self, c1, c2, k, s=1, g=1, bias=True, w=None):
        """Initializes a TensorFlow 2D convolution layer, mimicking PyTorch's nn.Conv2D functionality for given filter
        sizes and stride.
        """
        super().__init__()
        assert g == 1, "TF v2.2 Conv2D does not support 'groups' argument"
        self.conv = keras.layers.Conv2D(
            filters=c2,
            kernel_size=k,
            strides=s,
            padding="VALID",
            use_bias=bias,
            kernel_initializer=keras.initializers.Constant(w.weight.permute(2, 3, 1, 0).numpy()),
            bias_initializer=keras.initializers.Constant(w.bias.numpy()) if bias else None,
        )

    def call(self, inputs):
        """Applies a convolution operation to the inputs and returns the result."""
        return self.conv(inputs)


class TFBottleneckCSP(keras.layers.Layer):
    """Implements a CSP bottleneck layer for TensorFlow models to enhance gradient flow and efficiency."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        """
        Initializes CSP bottleneck layer with specified channel sizes, count, shortcut option, groups, and expansion
        ratio.

        Inputs are ch_in, ch_out, number, shortcut, groups, expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv2d(c1, c_, 1, 1, bias=False, w=w.cv2)
        self.cv3 = TFConv2d(c_, c_, 1, 1, bias=False, w=w.cv3)
        self.cv4 = TFConv(2 * c_, c2, 1, 1, w=w.cv4)
        self.bn = TFBN(w.bn)
        self.act = lambda x: keras.activations.swish(x)
        self.m = keras.Sequential([TFBottleneck(c_, c_, shortcut, g, e=1.0, w=w.m[j]) for j in range(n)])

    def call(self, inputs):
        """Processes input through the model layers, concatenates, normalizes, activates, and reduces the output
        dimensions.
        """
        y1 = self.cv3(self.m(self.cv1(inputs)))
        y2 = self.cv2(inputs)
        return self.cv4(self.act(self.bn(tf.concat((y1, y2), axis=3))))


class TFC3(keras.layers.Layer):
    """CSP bottleneck layer with 3 convolutions for TensorFlow, supporting optional shortcuts and group convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        """
        Initializes CSP Bottleneck with 3 convolutions, supporting optional shortcuts and group convolutions.

        Inputs are ch_in, ch_out, number, shortcut, groups, expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c1, c_, 1, 1, w=w.cv2)
        self.cv3 = TFConv(2 * c_, c2, 1, 1, w=w.cv3)
        self.m = keras.Sequential([TFBottleneck(c_, c_, shortcut, g, e=1.0, w=w.m[j]) for j in range(n)])

    def call(self, inputs):
        """
        Processes input through a sequence of transformations for object detection (YOLOv5).

        See https://github.com/ultralytics/yolov5.
        """
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3))


class TFC3x(keras.layers.Layer):
    """A TensorFlow layer for enhanced feature extraction using cross-convolutions in object detection models."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        """
        Initializes layer with cross-convolutions for enhanced feature extraction in object detection models.

        Inputs are ch_in, ch_out, number, shortcut, groups, expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c1, c_, 1, 1, w=w.cv2)
        self.cv3 = TFConv(2 * c_, c2, 1, 1, w=w.cv3)
        self.m = keras.Sequential(
            [TFCrossConv(c_, c_, k=3, s=1, g=g, e=1.0, shortcut=shortcut, w=w.m[j]) for j in range(n)]
        )

    def call(self, inputs):
        """Processes input through cascaded convolutions and merges features, returning the final tensor output."""
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3))


class TFSPP(keras.layers.Layer):
    """Implements spatial pyramid pooling for YOLOv3-SPP with specific channels and kernel sizes."""

    def __init__(self, c1, c2, k=(5, 9, 13), w=None):
        """Initializes a YOLOv3-SPP layer with specific input/output channels and kernel sizes for pooling."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_ * (len(k) + 1), c2, 1, 1, w=w.cv2)
        self.m = [keras.layers.MaxPool2D(pool_size=x, strides=1, padding="SAME") for x in k]

    def call(self, inputs):
        """Processes input through two TFConv layers and concatenates with max-pooled outputs at intermediate stage."""
        x = self.cv1(inputs)
        return self.cv2(tf.concat([x] + [m(x) for m in self.m], 3))


class TFSPPF(keras.layers.Layer):
    """Implements a fast spatial pyramid pooling layer for TensorFlow with optimized feature extraction."""

    def __init__(self, c1, c2, k=5, w=None):
        """Initializes a fast spatial pyramid pooling layer with customizable in/out channels, kernel size, and
        weights.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_ * 4, c2, 1, 1, w=w.cv2)
        self.m = keras.layers.MaxPool2D(pool_size=k, strides=1, padding="SAME")

    def call(self, inputs):
        """Executes the model's forward pass, concatenating input features with three max-pooled versions before final
        convolution.
        """
        x = self.cv1(inputs)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(tf.concat([x, y1, y2, self.m(y2)], 3))


class TFDetect(keras.layers.Layer):
    """Implements YOLOv5 object detection layer in TensorFlow for predicting bounding boxes and class probabilities."""

    def __init__(self, nc=80, anchors=(), ch=(), imgsz=(640, 640), w=None):
        """Initializes YOLOv5 detection layer for TensorFlow with configurable classes, anchors, channels, and image
        size.
        """
        super().__init__()
        self.stride = tf.convert_to_tensor(w.stride.numpy(), dtype=tf.float32)
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [tf.zeros(1)] * self.nl  # init grid
        self.anchors = tf.convert_to_tensor(w.anchors.numpy(), dtype=tf.float32)
        self.anchor_grid = tf.reshape(self.anchors * tf.reshape(self.stride, [self.nl, 1, 1]), [self.nl, 1, -1, 1, 2])
        self.m = [TFConv2d(x, self.no * self.na, 1, w=w.m[i]) for i, x in enumerate(ch)]
        self.training = False  # set to False after building model
        self.imgsz = imgsz
        for i in range(self.nl):
            ny, nx = self.imgsz[0] // self.stride[i], self.imgsz[1] // self.stride[i]
            self.grid[i] = self._make_grid(nx, ny)

    def call(self, inputs):
        """Performs forward pass through the model layers to predict object bounding boxes and classifications."""
        z = []  # inference output
        x = []
        for i in range(self.nl):
            x.append(self.m[i](inputs[i]))
            # x(bs,20,20,255) to x(bs,3,20,20,85)
            ny, nx = self.imgsz[0] // self.stride[i], self.imgsz[1] // self.stride[i]
            x[i] = tf.reshape(x[i], [-1, ny * nx, self.na, self.no])

            if not self.training:  # inference
                y = x[i]
                grid = tf.transpose(self.grid[i], [0, 2, 1, 3]) - 0.5
                anchor_grid = tf.transpose(self.anchor_grid[i], [0, 2, 1, 3]) * 4
                xy = (tf.sigmoid(y[..., 0:2]) * 2 + grid) * self.stride[i]  # xy
                wh = tf.sigmoid(y[..., 2:4]) ** 2 * anchor_grid
                # Normalize xywh to 0-1 to reduce calibration error
                xy /= tf.constant([[self.imgsz[1], self.imgsz[0]]], dtype=tf.float32)
                wh /= tf.constant([[self.imgsz[1], self.imgsz[0]]], dtype=tf.float32)
                y = tf.concat([xy, wh, tf.sigmoid(y[..., 4 : 5 + self.nc]), y[..., 5 + self.nc :]], -1)
                z.append(tf.reshape(y, [-1, self.na * ny * nx, self.no]))

        return tf.transpose(x, [0, 2, 1, 3]) if self.training else (tf.concat(z, 1),)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        """Generates a 2D grid of coordinates in (x, y) format with shape [1, 1, ny*nx, 2]."""
        # return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
        xv, yv = tf.meshgrid(tf.range(nx), tf.range(ny))
        return tf.cast(tf.reshape(tf.stack([xv, yv], 2), [1, 1, ny * nx, 2]), dtype=tf.float32)


class TFSegment(TFDetect):
    """YOLOv5 segmentation head for TensorFlow, combining detection and segmentation."""

    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), imgsz=(640, 640), w=None):
        """Initializes YOLOv5 Segment head with specified channel depths, anchors, and input size for segmentation
        models.
        """
        super().__init__(nc, anchors, ch, imgsz, w)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = [TFConv2d(x, self.no * self.na, 1, w=w.m[i]) for i, x in enumerate(ch)]  # output conv
        self.proto = TFProto(ch[0], self.npr, self.nm, w=w.proto)  # protos
        self.detect = TFDetect.call

    def call(self, x):
        """Applies detection and proto layers on input, returning detections and optionally protos if training."""
        p = self.proto(x[0])
        # p = TFUpsample(None, scale_factor=4, mode='nearest')(self.proto(x[0]))  # (optional) full-size protos
        p = tf.transpose(p, [0, 3, 1, 2])  # from shape(1,160,160,32) to shape(1,32,160,160)
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p)


class TFProto(keras.layers.Layer):
    """Implements convolutional and upsampling layers for feature extraction in YOLOv5 segmentation."""

    def __init__(self, c1, c_=256, c2=32, w=None):
        """Initializes TFProto layer with convolutional and upsampling layers for feature extraction and
        transformation.
        """
        super().__init__()
        self.cv1 = TFConv(c1, c_, k=3, w=w.cv1)
        self.upsample = TFUpsample(None, scale_factor=2, mode="nearest")
        self.cv2 = TFConv(c_, c_, k=3, w=w.cv2)
        self.cv3 = TFConv(c_, c2, w=w.cv3)

    def call(self, inputs):
        """Performs forward pass through the model, applying convolutions and upscaling on input tensor."""
        return self.cv3(self.cv2(self.upsample(self.cv1(inputs))))


class TFUpsample(keras.layers.Layer):
    """Implements a TensorFlow upsampling layer with specified size, scale factor, and interpolation mode."""

    def __init__(self, size, scale_factor, mode, w=None):
        """
        Initializes a TensorFlow upsampling layer with specified size, scale_factor, and mode, ensuring scale_factor is
        even.

        Warning: all arguments needed including 'w'
        """
        super().__init__()
        assert scale_factor % 2 == 0, "scale_factor must be multiple of 2"
        self.upsample = lambda x: tf.image.resize(x, (x.shape[1] * scale_factor, x.shape[2] * scale_factor), mode)
        # self.upsample = keras.layers.UpSampling2D(size=scale_factor, interpolation=mode)
        # with default arguments: align_corners=False, half_pixel_centers=False
        # self.upsample = lambda x: tf.raw_ops.ResizeNearestNeighbor(images=x,
        #                                                            size=(x.shape[1] * 2, x.shape[2] * 2))

    def call(self, inputs):
        """Applies upsample operation to inputs using nearest neighbor interpolation."""
        return self.upsample(inputs)


class TFConcat(keras.layers.Layer):
    """Implements TensorFlow's version of torch.concat() for concatenating tensors along the last dimension."""

    def __init__(self, dimension=1, w=None):
        """Initializes a TensorFlow layer for NCHW to NHWC concatenation, requiring dimension=1."""
        super().__init__()
        assert dimension == 1, "convert only NCHW to NHWC concat"
        self.d = 3

    def call(self, inputs):
        """Concatenates a list of tensors along the last dimension, used for NCHW to NHWC conversion."""
        return tf.concat(inputs, self.d)


def parse_model(d, ch, model, imgsz):
    """Parses a model definition dict `d` to create YOLOv5 model layers, including dynamic channel adjustments."""
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, ch_mul = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
        d.get("channel_multiple"),
    )
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    if not ch_mul:
        ch_mul = 8

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m_str = m
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [
            nn.Conv2d,
            Conv,
            DWConv,
            DWConvTranspose2d,
            Bottleneck,
            SPP,
            SPPF,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3x,
        ]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, ch_mul) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3x]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[-1 if x == -1 else x + 1] for x in f)
        elif m in [Detect, Segment]:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, ch_mul)
            args.append(imgsz)
        else:
            c2 = ch[f]

        tf_m = eval("TF" + m_str.replace("nn.", ""))
        m_ = (
            keras.Sequential([tf_m(*args, w=model.model[i][j]) for j in range(n)])
            if n > 1
            else tf_m(*args, w=model.model[i])
        )  # module

        torch_m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in torch_m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f"{i:>3}{str(f):>18}{str(n):>3}{np:>10}  {t:<40}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return keras.Sequential(layers), sorted(save)


class TFModel:
    """Implements YOLOv5 model in TensorFlow, supporting TensorFlow, Keras, and TFLite formats for object detection."""

    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, model=None, imgsz=(640, 640)):
        """Initializes TF YOLOv5 model with specified configuration, channels, classes, model instance, and input
        size.
        """
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding {cfg} nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value
        self.model, self.savelist = parse_model(deepcopy(self.yaml), ch=[ch], model=model, imgsz=imgsz)

    def predict(
        self,
        inputs,
        tf_nms=False,
        agnostic_nms=False,
        topk_per_class=100,
        topk_all=100,
        iou_thres=0.45,
        conf_thres=0.25,
    ):
        """Runs inference on input data, with an option for TensorFlow NMS."""
        y = []  # outputs
        x = inputs
        for m in self.model.layers:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.savelist else None)  # save output

        # Add TensorFlow NMS
        if tf_nms:
            boxes = self._xywh2xyxy(x[0][..., :4])
            probs = x[0][:, :, 4:5]
            classes = x[0][:, :, 5:]
            scores = probs * classes
            if agnostic_nms:
                nms = AgnosticNMS()((boxes, classes, scores), topk_all, iou_thres, conf_thres)
            else:
                boxes = tf.expand_dims(boxes, 2)
                nms = tf.image.combined_non_max_suppression(
                    boxes, scores, topk_per_class, topk_all, iou_thres, conf_thres, clip_boxes=False
                )
            return (nms,)
        return x  # output [1,6300,85] = [xywh, conf, class0, class1, ...]
        # x = x[0]  # [x(1,6300,85), ...] to x(6300,85)
        # xywh = x[..., :4]  # x(6300,4) boxes
        # conf = x[..., 4:5]  # x(6300,1) confidences
        # cls = tf.reshape(tf.cast(tf.argmax(x[..., 5:], axis=1), tf.float32), (-1, 1))  # x(6300,1)  classes
        # return tf.concat([conf, cls, xywh], 1)

    @staticmethod
    def _xywh2xyxy(xywh):
        """Converts bounding box format from [x, y, w, h] to [x1, y1, x2, y2], where xy1=top-left and xy2=bottom-
        right.
        """
        x, y, w, h = tf.split(xywh, num_or_size_splits=4, axis=-1)
        return tf.concat([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)


class AgnosticNMS(keras.layers.Layer):
    """Performs agnostic non-maximum suppression (NMS) on detected objects using IoU and confidence thresholds."""

    def call(self, input, topk_all, iou_thres, conf_thres):
        """Performs agnostic NMS on input tensors using given thresholds and top-K selection."""
        return tf.map_fn(
            lambda x: self._nms(x, topk_all, iou_thres, conf_thres),
            input,
            fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.int32),
            name="agnostic_nms",
        )

    @staticmethod
    def _nms(x, topk_all=100, iou_thres=0.45, conf_thres=0.25):
        """Performs agnostic non-maximum suppression (NMS) on detected objects, filtering based on IoU and confidence
        thresholds.
        """
        boxes, classes, scores = x
        class_inds = tf.cast(tf.argmax(classes, axis=-1), tf.float32)
        scores_inp = tf.reduce_max(scores, -1)
        selected_inds = tf.image.non_max_suppression(
            boxes, scores_inp, max_output_size=topk_all, iou_threshold=iou_thres, score_threshold=conf_thres
        )
        selected_boxes = tf.gather(boxes, selected_inds)
        padded_boxes = tf.pad(
            selected_boxes,
            paddings=[[0, topk_all - tf.shape(selected_boxes)[0]], [0, 0]],
            mode="CONSTANT",
            constant_values=0.0,
        )
        selected_scores = tf.gather(scores_inp, selected_inds)
        padded_scores = tf.pad(
            selected_scores,
            paddings=[[0, topk_all - tf.shape(selected_boxes)[0]]],
            mode="CONSTANT",
            constant_values=-1.0,
        )
        selected_classes = tf.gather(class_inds, selected_inds)
        padded_classes = tf.pad(
            selected_classes,
            paddings=[[0, topk_all - tf.shape(selected_boxes)[0]]],
            mode="CONSTANT",
            constant_values=-1.0,
        )
        valid_detections = tf.shape(selected_inds)[0]
        return padded_boxes, padded_scores, padded_classes, valid_detections


def activations(act=nn.SiLU):
    """Converts PyTorch activations to TensorFlow equivalents, supporting LeakyReLU, Hardswish, and SiLU/Swish."""
    if isinstance(act, nn.LeakyReLU):
        return lambda x: keras.activations.relu(x, alpha=0.1)
    elif isinstance(act, nn.Hardswish):
        return lambda x: x * tf.nn.relu6(x + 3) * 0.166666667
    elif isinstance(act, (nn.SiLU, SiLU)):
        return lambda x: keras.activations.swish(x)
    else:
        raise Exception(f"no matching TensorFlow activation found for PyTorch activation {act}")


def representative_dataset_gen(dataset, ncalib=100):
    """Generates a representative dataset for calibration by yielding transformed numpy arrays from the input
    dataset.
    """
    for n, (path, img, im0s, vid_cap, string) in enumerate(dataset):
        im = np.transpose(img, [1, 2, 0])
        im = np.expand_dims(im, axis=0).astype(np.float32)
        im /= 255
        yield [im]
        if n >= ncalib:
            break


def run(
    weights=ROOT / "yolov5s.pt",  # weights path
    imgsz=(640, 640),  # inference size h,w
    batch_size=1,  # batch size
    dynamic=False,  # dynamic batch size
):
    # PyTorch model
    """Exports YOLOv5 model from PyTorch to TensorFlow and Keras formats, performing inference for validation."""
    im = torch.zeros((batch_size, 3, *imgsz))  # BCHW image
    model = attempt_load(weights, device=torch.device("cpu"), inplace=True, fuse=False)
    _ = model(im)  # inference
    model.info()

    # TensorFlow model
    im = tf.zeros((batch_size, *imgsz, 3))  # BHWC image
    tf_model = TFModel(cfg=model.yaml, model=model, nc=model.nc, imgsz=imgsz)
    _ = tf_model.predict(im)  # inference

    # Keras model
    im = keras.Input(shape=(*imgsz, 3), batch_size=None if dynamic else batch_size)
    keras_model = keras.Model(inputs=im, outputs=tf_model.predict(im))
    keras_model.summary()

    LOGGER.info("PyTorch, TensorFlow and Keras models successfully verified.\nUse export.py for TF model export.")


def parse_opt():
    """Parses and returns command-line options for model inference, including weights path, image size, batch size, and
    dynamic batching.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="weights path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--dynamic", action="store_true", help="dynamic batch size")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes the YOLOv5 model run function with parsed command line options."""
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
