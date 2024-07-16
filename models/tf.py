# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
TensorFlow, Keras and TFLite versions of YOLOv5
Authored by https://github.com/zldrobit in PR https://github.com/ultralytics/yolov5/pull/1127

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
    # TensorFlow BatchNormalization wrapper
    def __init__(self, w=None):
        """
        Initialize a TensorFlow BatchNormalization layer using optional PyTorch pretrained weights.

        Args:
            w (torch.nn.Module | None): A PyTorch BatchNorm2d layer whose weights are used to initialize the TensorFlow
                BatchNormalization layer. If `None`, the layer is initialized with default parameters.

        Returns:
            (None): This constructor does not return anything.

        Example:
            ```python
            import torch
            from tensorflow.keras import layers

            # PyTorch BatchNorm2d layer
            torch_bn_layer = torch.nn.BatchNorm2d(32)

            # TensorFlow BatchNormalization layer with weights from PyTorch layer
            tf_bn_layer = TFBN(w=torch_bn_layer)
            ```
        """
        super().__init__()
        self.bn = keras.layers.BatchNormalization(
            beta_initializer=keras.initializers.Constant(w.bias.numpy()),
            gamma_initializer=keras.initializers.Constant(w.weight.numpy()),
            moving_mean_initializer=keras.initializers.Constant(w.running_mean.numpy()),
            moving_variance_initializer=keras.initializers.Constant(w.running_var.numpy()),
            epsilon=w.eps,
        )

    def call(self, inputs):
        """
        Apply batch normalization using TensorFlow's BatchNormalization layer.

        Args:
            inputs (torch.Tensor | np.ndarray | tf.Tensor): Input tensor to which batch normalization should be applied.
                The input tensor must have a compatible shape (N, C, H, W) or (N, H, W, C) depending on the framework.

        Returns:
            (tf.Tensor): Tensor after applying batch normalization, with the same shape as the input tensor.

        Example:
            ```python
            import torch
            from models.tf import TFBN

            # Create dummy input tensor
            input_tensor = torch.rand(1, 3, 64, 64)

            # Initialize TFBN layer
            tfbn = TFBN(w=None)  # 'w' is typically a pretrained layer, None is for demo

            # Apply batch normalization
            output_tensor = tfbn.call(input_tensor)
            ```
        """
        return self.bn(inputs)


class TFPad(keras.layers.Layer):
    # Pad inputs in spatial dimensions 1 and 2
    def __init__(self, pad):
        """
        Initialize a padding layer for spatial dimensions with specified padding, supporting both int and tuple inputs.

        Args:
            pad (int | tuple[int, int]): Padding size. If an integer is provided, the same padding will be applied
                to all sides. If a tuple, it should specify the (pad_height, pad_width).

        Returns:
            None

        Example:
            ```python
            # Example of initializing TFPad with integer padding
            pad_layer = TFPad(2)

            # Example of initializing TFPad with tuple padding
            pad_layer = TFPad((2, 3))
            ```

        Notes:
            - This padding layer will only affect the spatial dimensions (height and width) of the input tensor.
            - The `pad` parameter can either be a single integer or a tuple of two integers.
            - The shape of the padding tensor is [batch, pad_height, pad_width, channels].
        """
        super().__init__()
        if isinstance(pad, int):
            self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
        else:  # tuple/list
            self.pad = tf.constant([[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]])

    def call(self, inputs):
        """
        Apply zero-padding to the input tensor in the spatial dimensions 1 and 2 as specified.

        Args:
            inputs (tf.Tensor): Input tensor to be padded, with shape (N, H, W, C) where N is batch size, H is height,
                W is width, and C is the number of channels.

        Returns:
            (tf.Tensor): Padded tensor with the same type as the input tensor, having the shape
                [batch, padded_height, padded_width, channels], where padded_height and padded_width include
                the applied padding in the respective dimensions.

        Example:
            ```python
            import tensorflow as tf
            from tfpad import TFPad

            # Initialize padding layer with integer padding
            pad_layer = TFPad(2)

            # Create a sample input tensor with shape [1, 3, 3, 1]
            inputs = tf.ones([1, 3, 3, 1])

            # Apply padding
            padded_output = pad_layer.call(inputs)

            print(padded_output.shape)  # Output shape should be [1, 7, 7, 1] after padding
            ```

        Notes:
            - The padding is symmetric on each border of the height and width dimensions.
            - The input tensor's shape is preserved except for the spatial dimensions where padding is applied.
        """
        return tf.pad(inputs, self.pad, mode="constant", constant_values=0)


class TFConv(keras.layers.Layer):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):
        """
        Initializes a standard convolution layer with optional batch normalization and activation; supports only
        group=1.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int, optional): Kernel size. Defaults to 1.
            s (int, optional): Stride size. Defaults to 1.
            p (int | None, optional): Padding size. If None, it is automatically determined using `autopad` function.
                Defaults to None.
            g (int, optional): Number of groups. Only supports group=1. Defaults to 1.
            act (bool, optional): Whether to include activation. Defaults to True.
            w (torch.nn.Module | None, optional): Pretrained weights. If provided, these weights will be used to
                initialize the convolution layer and batch normalization layer (if present). Defaults to None.

        Returns:
            None

        Note:
            TensorFlow Conv2D does not support the 'groups' argument when using versions prior to 2.2.
            See https://stackoverflow.com/questions/52975843/comparing-conv2d-with-padding-between-tensorflow
            -and-pytorch for convolution padding inconsistencies between TensorFlow and PyTorch.
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
        """
        Apply convolution, batch normalization, and activation functions to input tensors in sequence.

        Args:
            inputs (tf.Tensor): Input tensor with shape (N, H, W, C) where N is the batch size, H is the height, W is
                the width, and C is the number of channels.

        Returns:
            (tf.Tensor): Transformed tensor post convolution, batch normalization, and activation function, with shape
                typically modified based on kernel size, stride, and padding.

        Example:
            ```python
            # Example usage of TFConv
            conv_layer = TFConv(c1=3, c2=16, k=3, s=1, act=True)
            output = conv_layer(tf.random.uniform([1, 64, 64, 3]))  # Output tensor after convolution operations
            ```

        Note:
            This function performs operations sequentially: convolution, batch normalization (if specified), and
            activation. Ensure input tensor dimensions match expected shape for compatibility.
        """
        return self.act(self.bn(self.conv(inputs)))


class TFDWConv(keras.layers.Layer):
    # Depthwise convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True, w=None):
        """
        Initialize a depthwise convolution layer with optional batch normalization and activation for TensorFlow models.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels. Must be a multiple of `c1`.
            k (int): Kernel size. Defaults to 1.
            s (int): Stride value. Defaults to 1.
            p (int | tuple[int, int] | None): Padding size. Defaults to None, which means auto padding is applied.
            act (bool): Whether to apply an activation function. Defaults to True.
            w (torch.nn.Module | None): Pretrained weights. Defaults to None.

        Returns:
            (None): This initializer does not return a value. It configures the layer properties.

        Example:
            ```python
            tfdwconv_layer = TFDWConv(c1=32, c2=64, k=3, s=1, p=1, act=True, w=pretrained_weights)
            ```

        Note:
            TensorFlow's depthwise convolution implementation requires `c2` to be a multiple of `c1`. Ensure that
            this condition is met when specifying the number of output channels.
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
        """
        Applies depthwise convolution, batch normalization, and activation to input tensors in TensorFlow models.

        Args:
            inputs (tf.Tensor): Input tensor to the layer, with shape (N, H, W, C) where N is the batch size, H is height,
                W is width, and C is the number of channels.

        Returns:
            (tf.Tensor): Processed tensor after applying depthwise convolution, batch normalization, and activation function,
                with appropriate padding if necessary.

        Example:
            ```python
            import tensorflow as tf
            from pretrained_weights import get_weights  # hypothetical function to get pretrained weights

            input_tensor = tf.random.normal([1, 224, 224, 32])  # Example input tensor

            # Assuming predefined weights
            pretrained_weights = get_weights('path/to/weights.pth')

            depthwise_conv_layer = TFDWConv(c1=32, c2=64, k=3, s=1, p=1, act=True, w=pretrained_weights)
            output_tensor = depthwise_conv_layer(input_tensor)
            print(output_tensor.shape)  # Should output the shape after application of the layer
            ```

        Notes:
            - Ensure `c2` is a multiple of `c1` to satisfy TensorFlow's depthwise convolution requirements.
            - Padding is handled to ensure the input and output dimensions align as per specifications.
        """
        return self.act(self.bn(self.conv(inputs)))


class TFDWConvTranspose2d(keras.layers.Layer):
    # Depthwise ConvTranspose2d
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0, w=None):
        """
        Initialize depthwise ConvTranspose2D layer with specific channel, kernel, stride, and padding settings.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels, which must be equal to `c1`.
            k (int): Kernel size, must be 4.
            s (int): Stride size.
            p1 (int): Padding size, must be 1.
            p2 (int): Output padding size for the transposed convolution.
            w (torch.nn.Module): PyTorch layer containing pretrained weights.

        Returns:
            (None): Initializes the depthwise ConvTranspose2D layer with the specified parameters.

        Example:
            ```python
            # Assuming `w` contains pretrained weights compatible with the layer
            trans_conv_layer = TFDWConvTranspose2d(c1=64, c2=64, k=4, s=2, p1=1, p2=0, w=some_pretrained_layer)
            ```

        Notes:
            - The kernel size `k` must be 4, and padding `p1` must be 1 for the layer to function correctly.
            - The number of input channels `c1` must equal the number of output channels `c2`.
            - Ensure the provided weights `w` match the layer's specifications before initialization.
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
        """
        Apply a depthwise transposed convolution to input tensors in TensorFlow models, concatenating outputs.

        Args:
            inputs (tf.Tensor): The input tensor with shape (N, H, W, C) where N is batch size, H is height, W is width,
                and C is the number of input channels.

        Returns:
            (tf.Tensor): Processed tensor with dimensions affected by the transposed convolution operation and concatenation
                along the channel dimension.

        Example:
            ```python
            inputs = tf.random.normal([1, 32, 32, 64])  # Example input tensor
            trans_conv_layer = TFDWConvTranspose2d(c1=64, c2=64, k=4, s=2, p1=1, p2=0, w=pretrained_weights)
            output = trans_conv_layer(inputs)  # Output tensor after transposed convolution
            ```

        Note:
            The input tensor must have the same number of input and output channels. The kernel size (k) must be 4 and
            the padding (p1) must be 1 to match the specific configuration supported by the module.
        """
        return tf.concat([m(x) for m, x in zip(self.conv, tf.split(inputs, self.c1, 3))], 3)[:, 1:-1, 1:-1]


class TFFocus(keras.layers.Layer):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):
        """
        Initialize `TFFocus` layer to focus width and height information into channel space.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int, optional): Kernel size for the convolution. Default is 1.
            s (int, optional): Stride for the convolution. Default is 1.
            p (int | tuple[int, int], optional): Padding for the convolution. Default is None.
            g (int, optional): Number of groups for the convolution (must be 1). Default is 1.
            act (bool, optional): Whether to use activation function. Default is True.
            w (torch.nn.Module, optional): Pretrained weights for the convolution. Default is None.

        Returns:
            None

        Example:
            ```python
            tf_focus = TFFocus(c1=64, c2=128, k=3, s=1, p=1, act=True)
            output = tf_focus(input_tensor)
            ```

        Note:
            The `TFFocus` layer reduces the spatial dimensions by focusing width and height information into the channel
            space before further processing. This operation helps in reducing the spatial complexity while retaining
            essential features in the channel dimensions.
        """
        super().__init__()
        self.conv = TFConv(c1 * 4, c2, k, s, p, g, act, w.conv)

    def call(self, inputs):
        """
        Focus width and height information into channel space and apply convolution.

        Args:
            inputs (tf.Tensor): Input tensor of shape (B, W, H, C) where B is the batch size, W and H are the spatial
                dimensions, and C is the number of channels.

        Returns:
            (tf.Tensor): Output tensor after applying pixel shuffling and convolution, with shape (B, W/2, H/2, 4C).

        Example:
            ```python
            layer = TFFocus(c1=64, c2=128, k=1, s=1, p=None, g=1, act=True, w=None)
            input_tensor = tf.random.normal((1, 128, 128, 64))
            output_tensor = layer(input_tensor)
            ```

        Note:
            The input tensor is downsampled by a factor of 2 along the width and height dimensions and the number of channels is
            expanded by a factor of 4.
        """
        inputs = [inputs[:, ::2, ::2, :], inputs[:, 1::2, ::2, :], inputs[:, ::2, 1::2, :], inputs[:, 1::2, 1::2, :]]
        return self.conv(tf.concat(inputs, 3))


class TFBottleneck(keras.layers.Layer):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, w=None):
        """
        Perform non-maximum suppression (NMS) on prediction boxes.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            shortcut (bool): Whether to use a residual (shortcut) connection. Default is True.
            g (int): Number of groups for grouped convolution. Default is 1.
            e (float): Expansion ratio to calculate the number of hidden channels. Default is 0.5.
            w (torch.nn.Module | None): Pretrained weights for the PyTorch model, used to initialize TensorFlow layers.

        Returns:
            (None): This constructor does not return a value. It initializes the bottleneck layer parameters.

        Example:
            ```python
            # Example of initializing TFBottleneck with specified parameters
            bottleneck_layer = TFBottleneck(c1=128, c2=256, shortcut=True, g=1, e=0.5, w=pretrained_weights)
            ```
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_, c2, 3, 1, g=g, w=w.cv2)
        self.add = shortcut and c1 == c2

    def call(self, inputs):
        """
        Applies a bottleneck transformation with optional skip connection in TensorFlow models.

        Args:
            inputs (tf.Tensor): Input tensor with shape (B, H, W, C), where B is the batch size, H is the height,
                W is the width, and C is the number of channels.

        Returns:
            (tf.Tensor): Output tensor with bottleneck transformation applied, maintaining the same spatial dimensions
                but with possibly different number of channels C_out.

        Example:
            ```python
            bottleneck_layer = TFBottleneck(c1=64, c2=128, shortcut=True, g=1, e=0.5, w=pretrained_weights)
            output = bottleneck_layer(input_tensor)
            ```

        Note:
            The transformation includes two convolutional layers with optional ReLU activation and batch normalization.
            If `shortcut` is enabled and the input and output channels match, a skip connection adds the input directly
            to the output.
        """
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs))


class TFCrossConv(keras.layers.Layer):
    # Cross Convolution
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False, w=None):
        """
        Initialize a cross convolution layer with optional expansion, groups, and shortcut functionality.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int, optional): Kernel size for the convolution. Defaults to 3.
            s (int, optional): Stride for the convolution. Defaults to 1.
            g (int, optional): Number of groups for grouped convolution. Defaults to 1.
            e (float, optional): Expansion ratio to determine the hidden channels. Defaults to 1.0.
            shortcut (bool, optional): Whether to use a residual (shortcut) connection. Defaults to False.
            w (torch.nn.Module | None, optional): Pretrained PyTorch weights for the corresponding convolutional layers.
                Defaults to None.

        Returns:
            None

        Example:
            ```python
            tfxconv = TFCrossConv(c1=64, c2=128, k=3, s=1, g=1, e=1.0, shortcut=False, w=pretrained_weights)
            output = tfxconv(input_tensor)
            ```

        Note:
            - The cross convolution layer consists of two consecutive convolution operations with kernel shapes (1, k) and (k, 1).
            - The `shortcut` option adds a residual connection if input and output channels are equal.
            - The expansion ratio `e` controls the number of hidden channels: hidden channels = int(c2 * e).
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, (1, k), (1, s), w=w.cv1)
        self.cv2 = TFConv(c_, c2, (k, 1), (s, 1), g=g, w=w.cv2)
        self.add = shortcut and c1 == c2

    def call(self, inputs):
        """
        Performs cross convolution operations with optional shortcut connections in TensorFlow models.

        Args:
            inputs (tf.Tensor): Input tensor with shape (B, H, W, C), where B is the batch size, H is the height,
                W is the width, and C is the number of channels.

        Returns:
            (tf.Tensor): Output tensor after applying the cross convolution operations, with shape (B, H_out, W_out, C_out),
                where H_out and W_out may differ from H and W depending on the convolution parameters, and C_out is the number
                of output channels.

        Example:
            ```python
            tf_cross_conv = TFCrossConv(c1=64, c2=128, k=3, s=1, g=1, e=1.0, shortcut=True, w=pretrained_weights)
            output_tensor = tf_cross_conv(input_tensor)
            ```

        Note:
            - The cross convolution operation involves two separable convolutions: one with a kernel size of (1, k) and another
              with (k, 1).
            - If the shortcut connection is enabled (`shortcut=True`) and the input and output channels are equal (`c1 == c2`),
              the input tensor is added directly to the output after the convolution operations.
            - This layer allows flexible expansion ratios (`e`) and supports grouped convolutions (`g`).
        """
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs))


class TFConv2d(keras.layers.Layer):
    # Substitution for PyTorch nn.Conv2D
    def __init__(self, c1, c2, k, s=1, g=1, bias=True, w=None):
        """
        Initialize a TensorFlow 2D convolution layer as an equivalent replacement for PyTorch's nn.Conv2D, without group
        convolutions.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int | tuple[int, int]): Kernel size for the convolution.
            s (int, optional): Stride size for the convolution. Defaults to 1.
            g (int, optional): Number of blocked connections from input channels to output channels (groups). Must be 1 as
                TensorFlow Conv2D does not support groups. Defaults to 1.
            bias (bool, optional): Whether to include a bias term in the convolution. Defaults to True.
            w (torch.nn.Conv2d | None, optional): Weights from a pre-trained PyTorch Conv2d layer to initialize this layer.
                Defaults to None.

        Returns:
            (TFConv2d): A TensorFlow 2D convolution layer initialized with the specified parameters and pre-trained weights
                if provided.

        Example:
            ```python
            from tensorflow.keras import Model
            from tensorflow.keras.layers import Input

            input_layer = Input(shape=(224, 224, 3))
            conv_layer = TFConv2d(3, 16, 3)
            output = conv_layer(input_layer)
            model = Model(inputs=input_layer, outputs=output)
            ```

        Note:
            TensorFlow Conv2D does not support group convolutions. The layer parameters should ensure that the number of
            groups (`g`) is set to 1.
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
        """
        Perform convolution operation on input tensors, mimicking PyTorch's nn.Conv2d functionality.

        Args:
            inputs (tf.Tensor): Input tensor of shape (B, H, W, C), where B is batch size, H is height, W is width, and C is channels.

        Returns:
            (tf.Tensor): Output tensor resulting from convolution operation, maintaining dimensions (B, H, W, C).

        Example:
            ```python
            from tensorflow.keras import Model, Input
            from my_ultralytics_module import TFConv2d

            input_layer = Input(shape=(224, 224, 3))
            conv_layer = TFConv2d(3, 16, 3)
            output_tensor = conv_layer(input_layer)
            model = Model(inputs=input_layer, outputs=output_tensor)
            ```

        Notes:
            The convolution operation assumes 'VALID' padding, meaning no padding is added to the input tensor. Ensure input tensor
            dimensions and kernel size are appropriately configured for this setting.
        """
        return self.conv(inputs)


class TFBottleneckCSP(keras.layers.Layer):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        """
        Initialize a CSP bottleneck layer with specified parameters for TensorFlow models, supporting optional shortcut
        connections.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of Bottleneck layers to stack. Default is 1.
            shortcut (bool): If True, adds shortcut connections in bottleneck layers. Default is True.
            g (int): Number of groups for grouped convolutions. Default is 1.
            e (float): Expansion ratio for hidden layer dimensionality. Default is 0.5.
            w (object): Pretrained weights struct containing TensorFlow layer weights. Default is None.

        Returns:
            None

        Example:
            ```python
            csp_layer = TFBottleneckCSP(c1=64, c2=128, n=3, shortcut=True, g=1, e=0.5, w=pretrained_weights)
            output = csp_layer(input_tensor)
            ```

        Note:
            This layer combines Cross Stage Partial Networks (CSPNet) principles which split the feature map
            into two parts, enabling improved gradient flow and reduction in computation cost.
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
        """
        Combines CSP bottleneck layers and processes input tensors through convolution, bottleneck, and activation.

        Args:
            inputs (tf.Tensor): Input tensor to be processed with shape (N, H, W, C) where N is the batch size, H is height,
                W is width, and C is the number of channels.

        Returns:
            (tf.Tensor): Output tensor after combining CSP bottleneck layers, with shape (N, H, W, C_out).

        Example:
            ```python
            # Assuming `inputs` is a pre-existing tensor
            csp_layer = TFBottleneckCSP(c1=64, c2=128, n=3, shortcut=True, g=1, e=0.5, w=pretrained_weights)
            output = csp_layer(inputs)
            ```

        Note:
            This method concatenates the results from multiple bottleneck layers and applies normalization and activation
            functions to produce the final output.
        """
        y1 = self.cv3(self.m(self.cv1(inputs)))
        y2 = self.cv2(inputs)
        return self.cv4(self.act(self.bn(tf.concat((y1, y2), axis=3))))


class TFC3(keras.layers.Layer):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        """
        Initializes the CSP bottleneck layer with 3 convolutions, with optional shortcuts and group convolutions.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of bottleneck layers within the CSP block.
            shortcut (bool): If True, add a shortcut to the bottleneck layers. Default is True.
            g (int): Number of groups in group convolutions. Default is 1.
            e (float): Expansion ratio for hidden channels. Default is 0.5.
            w (object | None): Pre-trained weights for initializing the layer. If None, random initialization is used. Default
                is None.

        Returns:
            (None): This initializer does not return a value, it configures the layer properties.

        Example:
            ```python
            layer = TFC3(128, 256, n=3, shortcut=False, g=1, e=0.5, w=pretrained_weights)
            input_tensor = tf.random.normal((1, 128, 128, 128))  # example input tensor
            output_tensor = layer(input_tensor)  # forward pass
            ```
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c1, c_, 1, 1, w=w.cv2)
        self.cv3 = TFConv(2 * c_, c2, 1, 1, w=w.cv3)
        self.m = keras.Sequential([TFBottleneck(c_, c_, shortcut, g, e=1.0, w=w.m[j]) for j in range(n)])

    def call(self, inputs):
        """
        Perform forward pass through CSP Bottleneck with 3 convolutions for object detection.

        Args:
            inputs (tf.Tensor): Input tensor of shape (N, H, W, C) where N is the batch size, H is the height, W is
                the width, and C is the number of channels.

        Returns:
            (tf.Tensor): Output tensor after CSP Bottleneck processing, with shape dependent on the layer
                configuration, generally retaining the batch size N.

        Note:
            The implemented CSP (Cross Stage Partial) Bottleneck with 3 convolutions helps in reducing
            computational complexity while retaining essential features for object detection.

        Example:
            ```python
            layer = TFC3(c1=128, c2=256, n=3, shortcut=False, g=1, e=0.5, w=pretrained_weights)
            output_tensor = layer(input_tensor)
            ```

            Refer to the YOLOv5 repository for more details:
            https://github.com/ultralytics/yolov5
        """
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3))


class TFC3x(keras.layers.Layer):
    # 3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        """
        Summary:
        Initialize the TFC3x layer with cross-convolutions for enhanced feature extraction in deep learning models.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of sub-layers to stack within the module.
            shortcut (bool): Indicates whether to use residual connections across sub-layers. Defaults to True.
            g (int): Number of groups for grouped convolution. Defaults to 1.
            e (float): Expansion ratio for hidden channels relative to the output channels. Defaults to 0.5.
            w (nn.Module | None): Pretrained weights for initializing the layer if available. Defaults to None.
        
        Returns:
            None
        
        Example:
            ```python
            # Initialize TFC3x layer with 64 input channels, 128 output channels, stacking 3 sub-layers,
            # using residual connections, and optional pretrained weights.
            layer = TFC3x(c1=64, c2=128, n=3, shortcut=True, g=1, e=0.5, w=pretrained_weights)
            ```
        
        Note:
            The TFC3x layer is designed for advanced convolutional neural network (CNN) architectures, particularly for tasks
            requiring efficient and enhanced feature extraction, such as object detection. The cross-convolution mechanism
            employed within this layer aims to boost the model's representational capacity by incorporating unique feature
            interactions across multiple sub-layers. For further information, refer to the YOLOv5 repository at
            https://github.com/ultralytics/yolov5. This layer can be combined with other TensorFlow components to build custom
            CNN models.
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
        """
        Processes input through cascaded cross-convolutions and merges features for enhanced object detection.

        Args:
            inputs (tf.Tensor): Input tensor of shape (N, H, W, C) where N is the batch size, H and W are the spatial dimensions,
                and C is the number of input channels.

        Returns:
            (tf.Tensor): Output tensor after applying cross-convolutions and concatenation with shape (N, H, W, C_out) where
                C_out is the number of output channels, as configured during initialization.

        Example:
            ```python
            import tensorflow as tf

            # Example initialization of TFC3x layer
            layer = TFC3x(c1=64, c2=128, n=2, shortcut=True, g=1, e=0.5, w=None)
            input_tensor = tf.random.normal((1, 128, 128, 64))  # Create a random tensor as input
            output_tensor = layer(input_tensor)  # Process the input tensor through the TFC3x layer
            ```

        Notes:
            For details on the cross convolutional structure and its application, visit the YOLOv5 repository at
            https://github.com/ultralytics/yolov5.
        """
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3))


class TFSPP(keras.layers.Layer):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), w=None):
        """
        Initialize a Spatial Pyramid Pooling (SPP) layer for YOLOv3 with specified input/output channels and kernel
        sizes.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (tuple[int, int, int]): Kernel sizes for max pooling layers. Defaults to (5, 9, 13).
            w (object | None): Pre-trained weights for initializing the convolutional layers.

        Returns:
            None

        Example:
            ```python
            spp_layer = TFSPP(c1=1024, c2=512, k=(5, 9, 13), w=pretrained_weights)
            output = spp_layer(input_tensor)
            ```

        Note:
            - The SPP layer enhances the receptive field during feature extraction by applying multiple max-pooling operations with different kernel sizes.
            - For more information, refer to the YOLOv5 repository at https://github.com/ultralytics/yolov5.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_ * (len(k) + 1), c2, 1, 1, w=w.cv2)
        self.m = [keras.layers.MaxPool2D(pool_size=x, strides=1, padding="SAME") for x in k]

    def call(self, inputs):
        """
        Applies spatial pyramid pooling (SPP) to the input, enhancing receptive field.

        Args:
            inputs (tf.Tensor): Input tensor of shape (B, H, W, C), where B is the batch size, H is the height, W is the width,
                and C is the number of channels.

        Returns:
            (tf.Tensor): Output tensor after applying SPP, with shape (B, H, W, C_out), where C_out is the number of output
                channels.

        Example:
            ```python
            tf_spp_layer = TFSPP(c1=128, c2=256, k=(5, 9, 13))
            input_tensor = tf.random.normal((1, 64, 64, 128))
            output_tensor = tf_spp_layer(input_tensor)
            ```

        Note:
            This function enhances the receptive field by concatenating max-pooled output with original input, providing richer
            spatial context for object detection.
        """
        x = self.cv1(inputs)
        return self.cv2(tf.concat([x] + [m(x) for m in self.m], 3))


class TFSPPF(keras.layers.Layer):
    # Spatial pyramid pooling-Fast layer
    def __init__(self, c1, c2, k=5, w=None):
        """
        Initialize a fast spatial pyramid pooling layer for TensorFlow, specifying the input and output channels, kernel
        size, and weights.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for the pooling layers.
            w (TFConv | None): Pretrained weights for the convolutional layers, if available.

        Returns:
            None: Initializes the TFSPPF layer, ready for use in forward passes for YOLO models.

        Example:
            ```python
            c1, c2, k = 512, 1024, 5
            sppf_layer = TFSPPF(c1, c2, k)
            input_tensor = tf.random.normal((1, 64, 64, 512))  # Example input tensor
            output_tensor = sppf_layer(input_tensor)
            ```
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_ * 4, c2, 1, 1, w=w.cv2)
        self.m = keras.layers.MaxPool2D(pool_size=k, strides=1, padding="SAME")

    def call(self, inputs):
        """
        Perform forward pass using fast spatial pyramid pooling (SPPF) layer on input tensors for feature extraction.

        Args:
            inputs (tf.Tensor): Input tensor with shape (B, H, W, C) representing the batch size, height, width, and channels
                of the images.

        Returns:
            (tf.Tensor): Output tensor with shape (B, H, W, C_out) after concatenation of max-pooled features and final
                convolution.

        Example:
            ```python
            inputs = tf.random.normal((1, 256, 256, 64))
            sppf = TFSPPF(64, 128, k=5)
            outputs = sppf(inputs)
            print(outputs.shape)  # Example output shape: (1, 256, 256, 128)
            ```

        Note:
            This layer is designed for efficient spatial pyramid pooling by concatenating features from multiple receptive
            fields, enhancing the network's understanding of spatial hierarchies.
        """
        x = self.cv1(inputs)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(tf.concat([x, y1, y2, self.m(y2)], 3))


class TFDetect(keras.layers.Layer):
    # TF YOLOv5 Detect layer
    def __init__(self, nc=80, anchors=(), ch=(), imgsz=(640, 640), w=None):
        """
        Initialize the TensorFlow YOLOv5 detection layer with specified parameters for classes, anchors, channels, and
        image size.

        Args:
            nc (int): Number of classes for detection.
            anchors (tuple): Anchor boxes defined as a tuple of tuples, e.g., ((10, 13), (16, 30), (33, 23)).
            ch (tuple): Number of channels for each detection layer.
            imgsz (tuple): Image size as a tuple (height, width), e.g., (640, 640).
            w (nn.Module): Pretrained PyTorch model weights used to initialize the TensorFlow layer.

        Returns:
            None

        Example:
            ```python
            anchors = ((10, 13), (16, 30), (33, 23))
            ch = (128, 256, 512)
            imgsz = (640, 640)
            nc = 80  # Number of classes for detection

            yolo_layer = TFDetect(nc=nc, anchors=anchors, ch=ch, imgsz=imgsz, w=pretrained_model)
            ```

        Note:
            Ensure that the pretrained PyTorch weights (`w`) are correctly converted for TensorFlow initialization. The model
            should be set to evaluation mode if not in training.
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
        """
        Applies detection layer forward pass to perform object detection in TensorFlow models.

        Args:
            inputs (list[tf.Tensor]): List of input tensors for each detection layer, with shape (B, H, W, C).
                Each tensor corresponds to a different detection layer, where B is the batch size, H and W are
                height and width, and C is the channel number.

        Returns:
            (list[tf.Tensor]): List of outputs for detected bounding boxes and class scores, each having shape
                (B, N, A * (5 + nc)) where N = H * W is the number of grid cells, A is the number of anchors,
                and nc is the number of classes.

        Example:
            ```python
            inputs = [tf.random.normal((1, 20, 20, 255)), tf.random.normal((1, 40, 40, 255))]
            detection_layer = TFDetect(nc=80, anchors=[(10,13), (16,30), (33,23)], ch=(255, 255), imgsz=(640, 640))
            outputs = detection_layer.call(inputs)
            for out in outputs:
                print(out.shape)
            ```

        Note:
            This function will only execute the inference path and will not perform training.
            For more details on the YOLOv5 model implementation, visit https://github.com/ultralytics/yolov5.
        """
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
        """
        Generate a 2D grid of coordinates for given dimensions (nx, ny).

        Args:
            nx (int): Number of grid columns.
            ny (int): Number of grid rows.

        Returns:
            (tf.Tensor): A tensor of shape (1, ny, nx, 2) representing the grid coordinates,
                         where the last dimension contains the (x, y) coordinates for each grid point.

        Example:
            ```python
            grid = TFDetect._make_grid(nx=3, ny=2)
            print(grid.numpy())
            # Output:
            # [[[[0 0]
            #    [1 0]
            #    [2 0]]
            #
            #   [[0 1]
            #    [1 1]
            #    [2 1]]]]
            ```
        Note:
            The returned grid tensor is useful for aligning bounding box predictions
            with the spatial dimensions of the feature map.
        """
        # return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
        xv, yv = tf.meshgrid(tf.range(nx), tf.range(ny))
        return tf.cast(tf.reshape(tf.stack([xv, yv], 2), [1, 1, ny * nx, 2]), dtype=tf.float32)


class TFSegment(TFDetect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), imgsz=(640, 640), w=None):
        """
        Initialize YOLOv5 Segment head with specified channel depths, anchors, and input size for segmentation models.

        Args:
            nc (int): Number of classes to detect.
            anchors (tuple): Anchors used for YOLO detection, defined as a tuple of tuple(s) with dimensions
                (anchor_count, 2).
            nm (int): Number of masks for segmentation.
            npr (int): Number of prototypes for mask generation.
            ch (tuple[int]): Tuple containing the number of input channels for each detection layer.
            imgsz (tuple[int, int]): Input image size in the format (height, width).
            w (object): Weights object containing predetermined network weights.

        Returns:
            None

        Example:
            ```python
            segment = TFSegment(nc=80, anchors=((10, 13), (16, 30), (33, 23)), nm=32, npr=256,
                                ch=(256, 512, 1024), imgsz=(640, 640), w=pretrained_weights)
            ```

        Note:
            This class inherits from `TFDetect` and extends it for segmentation tasks, adding mask prediction capabilities.
            The initialization includes setting up mask-related parameters and an additional proto layer for mask prediction.
        """
        super().__init__(nc, anchors, ch, imgsz, w)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = [TFConv2d(x, self.no * self.na, 1, w=w.m[i]) for i, x in enumerate(ch)]  # output conv
        self.proto = TFProto(ch[0], self.npr, self.nm, w=w.proto)  # protos
        self.detect = TFDetect.call

    def call(self, x):
        """
        Apply YOLOv5 segmentation head, including detection and prototype layers, to the input tensor.

        Args:
            x (list[tf.Tensor]): List of input tensors from the previous layer, with each tensor in the list having shape
                (B, H, W, C), where B is the batch size, H and W are the height and width, and C is the number of channels.

        Returns:
            (tuple[tf.Tensor, tf.Tensor]): A tuple containing:
                - Detection tensor with shape (N, A, G, no), where N is batch size, A is number of anchors, G is grid size,
                    and `no` is the number of outputs per anchor (no = 5 + number of classes + number of masks).
                - Prototype tensor with shape (B, nm, G, G) for segmentation masks where nm is the number of masks and
                    G is the grid size.

        Example:
            ```python
            segment = TFSegment(nc=80, anchors=((10, 13), (16, 30), (33, 23)), nm=32, npr=256, ch=(256, 512, 1024),
                                imgsz=(640, 640), w=pretrained_weights)
            detection, prototype = segment.call([input_tensor_1, input_tensor_2, input_tensor_3])
            ```

        Note:
            This method overrides the `call` method of `TFDetect` class to include a prototype layer for mask prediction.
            The output includes both detection and mask predictions, with masks being generated from the prototype tensor
            processed through the network.
        """
        p = self.proto(x[0])
        # p = TFUpsample(None, scale_factor=4, mode='nearest')(self.proto(x[0]))  # (optional) full-size protos
        p = tf.transpose(p, [0, 3, 1, 2])  # from shape(1,160,160,32) to shape(1,32,160,160)
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p)


class TFProto(keras.layers.Layer):
    def __init__(self, c1, c_=256, c2=32, w=None):
        """
        Initialize TFProto layer composed of convolutional and upsampling layers for feature extraction and
        transformation.

        Args:
            c1 (int): Number of input channels.
            c_ (int, optional): Number of intermediate channels. Defaults to 256.
            c2 (int, optional): Number of output channels. Defaults to 32.
            w (torch.nn.Module | None): Pretrained weights to initialize the TensorFlow convolution layers.

        Note:
            This layer is utilized within the YOLOv5 model for processing segmentation-specific features.
        """
        super().__init__()
        self.cv1 = TFConv(c1, c_, k=3, w=w.cv1)
        self.upsample = TFUpsample(None, scale_factor=2, mode="nearest")
        self.cv2 = TFConv(c_, c_, k=3, w=w.cv2)
        self.cv3 = TFConv(c_, c2, w=w.cv3)

    def call(self, inputs):
        """Performs forward pass through TFProto model applying convolutions and upscaling on input tensor."""
        return self.cv3(self.cv2(self.upsample(self.cv1(inputs))))


class TFUpsample(keras.layers.Layer):
    # TF version of torch.nn.Upsample()
    def __init__(self, size, scale_factor, mode, w=None):
        """
        Initialize a TensorFlow upsampling layer with specified size, scale factor, and mode.

        Args:
            size (tuple | None): Desired output size of the upsampled tensor. Ignored if `scale_factor` is specified.
            scale_factor (float | None): Multiplier for the spatial size of the input tensor. Must be an even number.
            mode (str): Upsampling algorithm. Supported modes include 'nearest' and 'bilinear'.
            w (None): Placeholder for weights, included for parameter consistency across layers.

        Returns:
            (None): This is an initializer method; thus, it does not return any value.

        Example:
            ```python
            upsample_layer = TFUpsample(size=None, scale_factor=2, mode='nearest')
            input_tensor = tf.random.normal([1, 64, 32, 32])  # Random input tensor of shape (B, C, H, W)
            output_tensor = upsample_layer(input_tensor)
            ```

        Note:
            The `scale_factor` must be an even number, ensuring proper scaling of tensor dimensions during upsampling.
            The default algorithm for upsampling is 'nearest', which utilizes nearest-neighbor interpolation.
        """
        super().__init__()
        assert scale_factor % 2 == 0, "scale_factor must be multiple of 2"
        self.upsample = lambda x: tf.image.resize(x, (x.shape[1] * scale_factor, x.shape[2] * scale_factor), mode)
        # self.upsample = keras.layers.UpSampling2D(size=scale_factor, interpolation=mode)
        # with default arguments: align_corners=False, half_pixel_centers=False
        # self.upsample = lambda x: tf.raw_ops.ResizeNearestNeighbor(images=x,
        #                                                            size=(x.shape[1] * 2, x.shape[2] * 2))

    def call(self, inputs):
        """
        Upsamples the input tensor using nearest neighbor interpolation by a specified scale factor.

        Args:
            inputs (tf.Tensor): Input tensor to be upsampled, of shape (N, H, W, C) where N is the batch size, H and W
                are the height and width respectively, and C is the number of channels.

        Returns:
            (tf.Tensor): Upsampled tensor with dimensions (N, H * scale_factor, W * scale_factor, C).

        Example:
            ```python
            upsample_layer = TFUpsample(size=None, scale_factor=2, mode='nearest')
            input_tensor = tf.random.normal([1, 64, 32, 32])  # Random input tensor of shape (B, C, H, W)
            output_tensor = upsample_layer(input_tensor)
            ```

        Note:
            This implementation uses TensorFlow's `tf.image.resize` function to perform the nearest neighbor
            upsampling. Ensure that the `scale_factor` is an even number for proper upsampling.
        """
        return self.upsample(inputs)


class TFConcat(keras.layers.Layer):
    # TF version of torch.concat()
    def __init__(self, dimension=1, w=None):
        """
        Initializes a TensorFlow layer for concatenating tensors, converting from NCHW to NHWC format.

        Args:
            dimension (int): Dimension along which to concatenate. Must be 1 for converting NCHW to NHWC.
            w (None, optional): Placeholder for weights to maintain consistency with other layers.

        Returns:
            None

        Note:
            This layer is designed to handle tensor concatenation specifically for dimensions related to
            NCHW (channels-first) to NHWC (channels-last) format conversion.

        Example:
            ```python
            concat_layer = TFConcat(dimension=1)
            tensor1 = tf.random.normal((1, 32, 224, 224))  # Example tensor in NCHW format
            tensor2 = tf.random.normal((1, 32, 224, 224))  # Another example tensor in NCHW format
            result = concat_layer([tensor1, tensor2])
            ```
        """
        super().__init__()
        assert dimension == 1, "convert only NCHW to NHWC concat"
        self.d = 3

    def call(self, inputs):
        """
        Concatenate input tensors along the last dimension for NCHW to NHWC conversion.

        Args:
            inputs (list[tf.Tensor]): List of input tensors, each with shape (B, H, W, C) where B is the batch size,
                H is height, W is width, and C is the number of channels.

        Returns:
            (tf.Tensor): Concatenated tensor along the last dimension, maintaining the shape (B, H, W, ∑C_in),
                where ∑C_in is the sum of the input channels.

        Example:
            ```python
            concat_layer = TFConcat()
            tensor1 = tf.random.normal([1, 64, 64, 32])
            tensor2 = tf.random.normal([1, 64, 64, 64])
            concatenated_tensor = concat_layer([tensor1, tensor2])
            ```

        Note:
            This function is designed to convert concatenation from PyTorch's NCHW format to TensorFlow's NHWC format.
            Ensure all input tensors have compatible shapes except for the channel dimension.
        """
        return tf.concat(inputs, self.d)


def parse_model(d, ch, model, imgsz):
    """
    Parse YOLOv5 model configuration for TensorFlow and create its layer structure.

    Args:
        d (dict): Model configuration dictionary containing backbone and head definitions.
        ch (list[int]): List of channel numbers for each layer.
        model (object): Instance of the model with weights.
        imgsz (tuple[int, int]): Input image size (height, width).

    Returns:
        (list[keras.Sequential]): List of Keras Sequential models implementing the YOLOv5 architecture as specified by the
        given configuration dictionary.

    Example:
        ```python
        config = {
            "anchors": [[(10, 13), (16, 30), (33, 23)]],
            "nc": 80,
            "depth_multiple": 0.33,
            "width_multiple": 0.50,
            "backbone": [
                [-1, 1, "Conv", [64, 3, 1, 1]],
                [-1, 1, "BottleneckCSP", [128, 1, True]],
                [-1, 1, "SPP", [256]],
            ],
            "head": [
                [-1, 1, "BottleneckCSP", [128, 1, True]],
                [-1, 1, "Detect", [80, [[10, 13], [16, 30], [33, 23]], 256]],
            ],
        }
        ch = [3]
        model = YourModel()  # Substitute with an actual model object
        imgsz = (640, 640)

        parsed_model = parse_model(config, ch, model, imgsz)
        ```

    Note:
        This function dynamically creates the model layers specified by the configuration dictionary, ensuring channel
        dimensions, kernel sizes, and other parameters match expected values. It supports various module types like
        `Conv`, `Bottleneck`, `SPP`, and more.
    """
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
    # TF YOLOv5 model
    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, model=None, imgsz=(640, 640)):
        """
        Initialize the TensorFlow YOLOv5 model instance based on given configuration, channel count, and optional pre-
        trained weights.

        Args:
            cfg (str | dict): Path to the YOLOv5 configuration file (YAML) or a dictionary containing the model definition.
            ch (int): Number of input channels, usually 3 for RGB images.
            nc (int | None): Number of classes for detection. If provided, this overrides the `nc` value in the configuration.
            model (torch.nn.Module | None): Pre-trained PyTorch model instance to convert its weights for TensorFlow model.
            imgsz (tuple[int, int]): Image dimensions specified as (height, width).

        Returns:
            (None)

        Example:
            ```python
            tf_model = TFModel(cfg='models/yolov5s.yaml', ch=3, nc=80, model=pytorch_model_instance, imgsz=(640, 640))
            ```

        Note:
            Ensure that the provided model configuration file or dictionary conforms to the expected YOLOv5 format for seamless
            loading and parsing. Refer to https://github.com/ultralytics/yolov5 for additional details and model configurations.
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
        """
        Predict bounding boxes and class scores for given input data using YOLOv5 model layers with optional TensorFlow
        NMS.

        Args:
            inputs (tf.Tensor): Input tensor containing the image data for object detection.
            tf_nms (bool): If True, apply TensorFlow NMS (Non-Maximum Suppression) to filter overlapping bounding boxes.
                Default is False.
            agnostic_nms (bool): If True, apply class-agnostic NMS. Default is False.
            topk_per_class (int): The maximum number of boxes retained per class after NMS. Default is 100.
            topk_all (int): The maximum number of boxes retained across all classes after NMS. Default is 100.
            iou_thres (float): Intersection-over-Union (IoU) threshold for filtering overlapping boxes in NMS. Default is 0.45.
            conf_thres (float): Confidence threshold for filtering low-confidence predictions. Default is 0.25.

        Returns:
            (tuple[tf.Tensor]): Predicted bounding boxes and class scores, optionally filtered by TensorFlow NMS.
                If `tf_nms` is True, the returned tuple will include NMS filtered boxes.

        Example:
            ```python
            import tensorflow as tf
            model = TFModel(cfg='yolov5s.yaml', model=pretrained_model, imgsz=(640, 640))

            # Generate dummy input data
            inputs = tf.random.normal((1, 640, 640, 3))

            # Run prediction
            outputs = model.predict(inputs, tf_nms=True)
            ```

        Note:
            The function supports both standard YOLOv5 inference and optional TensorFlow NMS for filtering predictions.
            Ensure input tensor dimensions align with model expectations to avoid runtime errors.
        """
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
        """
        Converts bounding box format from [x, y, w, h] to [x1, y1, x2, y2].

        Args:
            xywh (tf.Tensor): Bounding boxes in [x, y, w, h] format with shape (N, 4),
                where N is the number of bounding boxes.

        Returns:
            (tf.Tensor): Bounding boxes in [x1, y1, x2, y2] format with shape (N, 4), corresponding to top-left
                (x1, y1) and bottom-right (x2, y2) coordinates.

        Example:
            ```python
            boxes_xywh = tf.constant([[100, 150, 200, 250]], dtype=tf.float32)
            boxes_xyxy = TFModel._xywh2xyxy(boxes_xywh)
            print(boxes_xyxy)
            # Output: tf.Tensor([[0., 0., 1., 1.]], shape=(1, 4), dtype=float32)
            ```

        Note:
            This function is primarily used internally for post-processing the output of model predictions.
        """
        x, y, w, h = tf.split(xywh, num_or_size_splits=4, axis=-1)
        return tf.concat([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)


class AgnosticNMS(keras.layers.Layer):
    # TF Agnostic NMS
    def call(self, input, topk_all, iou_thres, conf_thres):
        """
        Perform class-agnostic non-maximum suppression (NMS) on predicted bounding boxes.

        Args:
            inputs (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple consisting of:
                - boxes (torch.Tensor): Tensor containing bounding box coordinates with shape (N, 4) for (x1, y1, x2, y2).
                - scores (torch.Tensor): Tensor containing class scores for each bounding box with shape (N, num_classes).
                - classes (torch.Tensor): Tensor containing class indices for each bounding box with shape (N,).
            max_detections (int): Maximum number of detections to keep after NMS.
            iou_threshold (float): IoU threshold for determining whether to suppress a bounding box.
            score_threshold (float): Minimum score for a box to be considered a valid detection.

        Returns:
            (list[torch.Tensor]): List of tensors post NMS processing, consisting of:
                - boxes (torch.Tensor): Tensor with shape (M, 4), where M≤max_detections after NMS.
                - scores (torch.Tensor): Tensor with shape (M,) containing the scores of the remaining boxes.
                - classes (torch.Tensor): Tensor with shape (M,) containing the class indices of the remaining boxes.

        Example:
            ```python
            boxes = torch.tensor([[100, 100, 200, 200], [110, 110, 210, 210], [300, 300, 400, 400]])
            scores = torch.tensor([0.9, 0.75, 0.8])
            classes = torch.tensor([1, 1, 2])
            nms = AgnosticNMS()
            kept_boxes, kept_scores, kept_classes = nms(boxes, scores, classes, max_detections=2, iou_threshold=0.5, score_threshold=0.3)
            ```

        Note:
            The function performs class-agnostic NMS, meaning that it treats all classes as a single class for the purpose of
            suppression, leading to a more aggressive filtering of overlapping boxes across different classes.
        """
        return tf.map_fn(
            lambda x: self._nms(x, topk_all, iou_thres, conf_thres),
            input,
            fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.int32),
            name="agnostic_nms",
        )

    @staticmethod
    def _nms(x, topk_all=100, iou_thres=0.45, conf_thres=0.25):
        """
        Perform non-maximum suppression (NMS) on detected objects using class-agnosticity.

        Args:
            x (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing:
                - boxes (torch.Tensor): Bounding boxes with shape (N, 4), where N is the number of boxes.
                - classes (torch.Tensor): Class scores with shape (N, C), where C is the number of classes.
                - scores (torch.Tensor): Detection scores with shape (N, 1).
            topk_all (int): Maximum number of detections to keep after NMS.
            iou_thres (float): Intersection over Union (IoU) threshold for NMS.
            conf_thres (float): Confidence score threshold for filtering predictions.

        Returns:
            (tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]): A tuple containing:
                - padded_boxes (torch.Tensor): Padded bounding boxes with shape (topk_all, 4).
                - padded_scores (torch.Tensor): Padded scores with shape (topk_all,).
                - padded_classes (torch.Tensor): Padded class indices with shape (topk_all,).
                - valid_detections (int): Number of valid detections after NMS.

        Example:
            ```python
            boxes = torch.tensor([[0, 0, 10, 10], [0, 0, 8, 8]], dtype=torch.float32)
            scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
            classes = torch.tensor([[1, 0], [1, 0]], dtype=torch.float32)
            results = AgnosticNMS._nms((boxes, classes, scores), topk_all=10, iou_thres=0.5, conf_thres=0.3)
            print(results)
            ```

        Note:
            This method computes agnostic NMS, meaning classes are not considered in the suppression process. Use defined IoU and
            confidence thresholds to adjust filtering as needed.
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
    """
    Provide an equivalent TensorFlow function/method when translating PyTorch code to TensorFlow.

    Args:
        act (torch.nn.Module): PyTorch activation module. Valid options include nn.LeakyReLU, nn.Hardswish, nn.SiLU, and
            custom SiLU from utils.activations.

    Returns:
        (tf.function): TensorFlow equivalent activation function based on the provided PyTorch activation module.

    Examples:
        ```python
        from tensorflow import keras

        # Using LeakyReLU activation
        activation_tf = activations(nn.LeakyReLU())
        output = activation_tf(input_tensor)

        # Using Hardswish activation
        activation_tf = activations(nn.Hardswish())
        output = activation_tf(input_tensor)
        ```

    Note:
        This mapping utility ensures compatibility when converting models across TensorFlow and PyTorch frameworks,
        allowing the use of equivalent activation functions based on the original PyTorch model configuration.
    """
    if isinstance(act, nn.LeakyReLU):
        return lambda x: keras.activations.relu(x, alpha=0.1)
    elif isinstance(act, nn.Hardswish):
        return lambda x: x * tf.nn.relu6(x + 3) * 0.166666667
    elif isinstance(act, (nn.SiLU, SiLU)):
        return lambda x: keras.activations.swish(x)
    else:
        raise Exception(f"no matching TensorFlow activation found for PyTorch activation {act}")


def representative_dataset_gen(dataset, ncalib=100):
    """
    Generate a representative dataset for TensorFlow Lite model calibration by yielding input tensors.

    Args:
        dataset (object): The dataset object providing image paths, images, original images, video captures, and
            strings. The dataset should support iteration yielding tuples (path, img, im0s, vid_cap, string).
        ncalib (int): The number of calibration samples to generate. Default is 100.

    Returns:
        generator: A generator yielding a list containing a single tensor with shape (1, H, W, C) per dataset sample.

    Example:
        ```python
        dataset = CustomDataset(...)  # Custom dataset
        representative_data = representative_dataset_gen(dataset, ncalib=50)
        for data in representative_data:
            print(data)
        ```

    Notes:
        - The function expects input images to be in (C, H, W) format; it then converts them to (H, W, C) and normalizes.
        - Ensure that dataset supports iteration and typical indexing to avoid iteration-related issues.
        - Refer to TensorFlow Lite documentation for details on the role of representative datasets during model
          quantization: https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization.
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
    """Def run(weights=ROOT / "yolov5s.pt", imgsz=(640, 640), batch_size=1, dynamic=False):"""
        Exports YOLOv5 model from PyTorch to TensorFlow/Keras formats and performs inference for validation.
    
        Args:
            weights (str | Path): Path to the pre-trained YOLOv5 weights file (typically a .pt file).
            imgsz (tuple[int, int]): Tuple specifying the inference size (height, width) of the input images.
            batch_size (int): Number of images to process in a batch.
            dynamic (bool): Specifies dynamic batch size when set to True.
    
        Returns:
            None: The function does not return any value. It displays model summaries and performs inference.
    
        Example:
            ```python
            run(weights="yolov5s.pt", imgsz=(640, 640), batch_size=1, dynamic=False)
            ```
    
        Note:
            - Ensure that the specified weight file path points to a valid YOLOv5 weight file, and the weights are compatible
              with the model configuration.
            - The function will load the PyTorch model, convert it to TensorFlow/Keras formats, and display the model
              summaries. It will also perform a dummy inference to validate the export.
        """
        # PyTorch model
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
    """
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
    """
    Parses and returns command-line options for model inference.

    This function sets up an argument parser for command-line options pertinent to performing model inference, including
    the weights path, image size, batch size, and dynamic batching.

    Args:
        None

    Returns:
        (argparse.Namespace): Parsed arguments as namespace with attributes 'weights' (str), 'imgsz' (list[int]),
            'batch_size' (int), and 'dynamic' (bool).

    Example:
        ```python
        opts = parse_opt()
        print(opts.weights)  # Outputs the path to the weights file
        ```

    See Also:
        - https://github.com/ultralytics/ultralytics for repository context and additional details.
        - `argparse` documentation: https://docs.python.org/3/library/argparse.html for more about argument parsing.
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
    """
    Main entry point for execution of TensorFlow YOLOv5 model conversion script.

    Args:
        opt (argparse.Namespace): Parsed command line options, including weights path, image size, batch size, and dynamic
            batch size flag.

    Returns:
        None
    """
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
