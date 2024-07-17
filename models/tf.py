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
        Initializes a TensorFlow BatchNormalization layer, optionally using pretrained weights for initialization.
        
        Args:
            w (torch.nn.Module | None): PyTorch BatchNormalization layer whose weights are used to initialize the TensorFlow 
                BatchNormalization layer. If None, the BatchNormalization layer is initialized with default parameters.
        
        Returns:
            (None): This constructor does not return any value.
        
        Example:
            ```python
            import torch.nn as nn
            from tensorflow.keras import layers
            
            # Create a PyTorch batch normalization layer
            torch_bn = nn.BatchNorm2d(num_features=64)
            
            # Initialize a TFBN layer with PyTorch BN weights
            tf_bn = TFBN(w=torch_bn)
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
        Apply batch normalization to the given inputs using pretrained weights.
        
        Args:
            inputs (tf.Tensor): Input tensor to normalize, with shape (batch_size, ..., channels).
        
        Returns:
            (tf.Tensor): Batch-normalized tensor with same shape as the input.
        
        Example:
            ```python
            # Assume `inputs` is a TensorFlow tensor with shape (N, H, W, C)
            bn_layer = TFBN(w=pretrained_weights)
            normalized_output = bn_layer.call(inputs)
            ```
        
        Note:
            The `w` parameter used during initialization must be a PyTorch BatchNorm layer containing 
            pretrained weights. Ensure the `w` object has `bias`, `weight`, `running_mean`, `running_var`, 
            and `eps` attributes used for initializing the TFBN layer.
        """
        return self.bn(inputs)


class TFPad(keras.layers.Layer):
    # Pad inputs in spatial dimensions 1 and 2
    def __init__(self, pad):
        """
        Initialize a padding layer for spatial dimensions 1 and 2.
        
        Args:
            pad (int | tuple[int, int]): Padding size for the spatial dimensions. If an integer is provided, the same
                padding is applied symmetrically to the spatial dimensions. If a tuple is provided, it should contain two
                integers representing padding for height and width respectively.
        
        Returns:
            None
        
        Example:
            ```python
            # Using integer padding
            padding_layer = TFPad(1)
        
            # Using tuple padding
            padding_layer = TFPad((1, 2))
            ```
        
        Note:
            The padding is added to the input tensor in TensorFlow format, i.e., [[0, 0], [pad_height, pad_height],
            [pad_width, pad_width], [0, 0]].
        """
        super().__init__()
        if isinstance(pad, int):
            self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
        else:  # tuple/list
            self.pad = tf.constant([[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]])

    def call(self, inputs):
        """
        Pad an input tensor with zeros in specified spatial dimensions.
        
        Args:
            inputs (tf.Tensor): Input tensor to be padded, with shape (N, H, W, C).
        
        Returns:
            (tf.Tensor): Padded tensor with shape (N, H + 2 * pad_height, W + 2 * pad_width, C).
        
        Example:
            ```python
            import tensorflow as tf
            from your_module import TFPad
        
            # Create a sample input tensor with shape (1, 5, 5, 1)
            input_tensor = tf.random.normal((1, 5, 5, 1))
        
            # Using integer padding
            padding_layer = TFPad(1)
            output_tensor = padding_layer.call(input_tensor)
        
            # Using tuple padding
            padding_layer = TFPad((1, 2))
            output_tensor = padding_layer.call(input_tensor)
            ```
        
        Note:
            The padding is added to the input tensor in TensorFlow format, i.e., [[0, 0], [pad_height, pad_height], 
            [pad_width, pad_width], [0, 0]].
        """
        return tf.pad(inputs, self.pad, mode="constant", constant_values=0)


class TFConv(keras.layers.Layer):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):
        """
        Performs a standard 2D convolution with optional batch normalization and activation in TensorFlow.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int, optional): Kernel size of the convolution. Default is 1.
            s (int, optional): Stride of the convolution. Default is 1.
            p (int | None, optional): Padding size. If None, padding is automatically determined. Default is None.
            g (int, optional): Number of groups for grouped convolution. Default is 1. Note: must be 1 for TF.
            act (bool, optional): Boolean to include activation. Default is True.
            w (torch.nn.Module | None, optional): Pretrained weights from a PyTorch model to initialize the layer. Default is None.
        
        Returns:
            None: This function initializes an instance of the TFConv class.
        
        Example:
            ```python
            tf_conv = TFConv(c1=32, c2=64, k=3, s=1, w=pretrained_weights)
            ```
        Note:
            TF v2.2 Conv2D does not support the 'groups' argument (must be 1).
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
        Apply convolution, batch normalization, and activation to input tensors.
        
        Args:
            inputs (tf.Tensor): Input tensor with shape (N, H, W, C) where N is the batch size, H is the height, 
                W is the width, and C is the number of channels.
        
        Returns:
            (tf.Tensor): Output tensor after applying convolution, batch normalization, and activation, 
                maintaining shape (N, H, W, C).
        
        Example:
            ```python
            input_tensor = tf.random.normal((1, 224, 224, 3))
            conv_layer = TFConv(c1=3, c2=16, k=3, s=1)
            output_tensor = conv_layer(input_tensor)
            ```
        
        Note:
            This method calls the `call` method of the internal sequential layers consisting of padding (if stride 
            isn't 1), convolution, batch normalization (if enabled), and activation function (if enabled).
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
            k (int, optional): Size of the convolution kernel. Default is 1.
            s (int, optional): Stride of the convolution. Default is 1.
            p (int | tuple[int, int] | None, optional): Padding size; supports both integer and tuple inputs. Default is None.
            act (bool, optional): Whether to apply an activation function. Default is True.
            w (object | None, optional): Pretrained weights. Default is None.
        
        Returns:
            (None): This constructor does not return any values.
        
        Example:
            ```python
            import keras
            from models.tf import TFDWConv
        
            # Initialize the layer
            conv_layer = TFDWConv(c1=32, c2=64, k=3, s=1, p=1, act=True, w=pretrained_weights)
            ```
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
        Applies depthwise convolution, batch normalization, and an activation function to the input tensors.
        
        Args:
            inputs (tf.Tensor): Input tensor with shape (N, H, W, C), representing a batch of images.
        
        Returns:
            (tf.Tensor): Resulting tensor after the depthwise convolution, batch normalization, and activation are applied, 
                with shape (N, H', W', C') depending on the convolution parameters.
        
        Example:
            ```python
            import tensorflow as tf
            from models.tf import TFDWConv
        
            # Dummy input tensor with shape (batch_size, height, width, channels)
            inputs = tf.random.normal([8, 32, 32, 32])
        
            # Initialize depthwise convolution layer
            conv_layer = TFDWConv(c1=32, c2=64, k=3, s=1, p=1, act=True)
        
            # Apply depthwise convolution
            outputs = conv_layer(inputs)
            ```
        
        Note:
            Padding is added to the input tensor in TensorFlow format, i.e., [[0, 0], [pad_height, pad_height], [pad_width, 
            pad_width], [0, 0]].
        """
        return self.act(self.bn(self.conv(inputs)))


class TFDWConvTranspose2d(keras.layers.Layer):
    # Depthwise ConvTranspose2d
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0, w=None):
        """
        Initialize a Depthwise ConvTranspose2D layer with specific channel, kernel, stride, and padding configurations.
        
        Args:
            c1 (int): Number of input channels; must equal `c2`.
            c2 (int): Number of output channels; must equal `c1`.
            k (int): Kernel size; currently supports only `k=4`.
            s (int): Stride size for the transposed convolution.
            p1 (int): Padding applied to the original input; currently supports only `p1=1`.
            p2 (int): Additional padding applied to the transposed output.
            w (torch.nn.Module): Pre-trained weights, including both kernel and bias, for initialization.
        
        Returns:
            (None): This constructor does not return any values.
        
        Example:
            ```python
            import tensorflow as tf
            from models.tf import TFDWConvTranspose2d
        
            # Define input tensor
            input_tensor = tf.random.normal([1, 64, 64, 32])
        
            # Initialize the TFDWConvTranspose2d layer
            depthwise_conv_transpose2d = TFDWConvTranspose2d(c1=32, c2=32, k=4, s=2, p1=1, p2=0, w=pretrained_weights)
        
            # Apply the layer
            output_tensor = depthwise_conv_transpose2d(input_tensor)
            ```
        
        Note:
            This layer is designed for depthwise convolution with specific constraints on kernel size and initial padding.
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
        Perform upsampling using depthwise transposed convolution, followed by concatenation across channels.
        
        Args:
            inputs (tf.Tensor): Input tensor with shape (N, H, W, C1), where N is batch size, H is height, W is width, 
                and C1 is the number of input channels.
        
        Returns:
            (tf.Tensor): Output tensor after applying depthwise ConvTranspose2D and concatenation, with shape 
                (N, (H-1)*stride + kernel_size, (W-1)*stride + kernel_size, C1). After upsampling, 1 pixel is cropped 
                from the border of the output to match expected dimensions.
        
        Example:
            ```python
            # Define input tensor
            input_tensor = tf.random.normal([1, 64, 64, 32])
            
            # Initialize the TFDWConvTranspose2d layer
            depthwise_conv_transpose2d_layer = TFDWConvTranspose2d(c1=32, c2=32, k=4, s=2, p1=1, p2=0, w=w)
        
            # Apply the layer
            output_tensor = depthwise_conv_transpose2d_layer(input_tensor)
            ```
        
        Note:
            This function handles specific kernel size (k=4) and padding constraints (p1=1) for depthwise ConvTranspose2D.
        """
        return tf.concat([m(x) for m, x in zip(self.conv, tf.split(inputs, self.c1, 3))], 3)[:, 1:-1, 1:-1]


class TFFocus(keras.layers.Layer):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):
        """
        Initializes TFFocus layer to focus width and height information into channel space with custom convolution parameters.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int, optional): Size of the convolutional kernel. Default is 1.
            s (int, optional): Stride value for the convolutional layer. Default is 1.
            p (int | None, optional): Padding value. If None, will be automatically determined based on `k`. Default is None.
            g (int, optional): Number of groups for the convolution. Default is 1.
            act (bool, optional): Whether to use an activation layer. Default is True.
            w (torch.nn.Module | None, optional): Pre-trained weight object containing convolution, batch norm, and activation
                layers. Default is None.
        
        Returns:
            None
        
        Example:
            ```python
            focus_layer = TFFocus(c1=3, c2=64, k=3, s=1, p=1, act=True)
            output = focus_layer(inputs)
            ```
        
        Note:
            Ensure that the input tensor dimensions match the expected values for width and height focusing to operate correctly.
        """
        super().__init__()
        self.conv = TFConv(c1 * 4, c2, k, s, p, g, act, w.conv)

    def call(self, inputs):
        """
        Perform pixel shuffling and convolution on the input tensor, converting spatial dimensions into channel space.
        
        Args:
            inputs (tf.Tensor): Input tensor with shape (B, H, W, C), where B is the batch size, H is the height, 
                W is the width, and C is the number of channels.
        
        Returns:
            (tf.Tensor): Output tensor after pixel shuffling and convolution, with shape (B, H/2, W/2, 4C).
        
        Example:
            ```python
            focus_layer = TFFocus(c1=32, c2=64, k=1, s=1)
            output = focus_layer(inputs)  # inputs should be a tensor with shape (B, H, W, C)
            ```
        
        Note:
            Ensure input tensor dimensions match expected values for correct width and height focusing.
        """
        inputs = [inputs[:, ::2, ::2, :], inputs[:, 1::2, ::2, :], inputs[:, ::2, 1::2, :], inputs[:, 1::2, 1::2, :]]
        return self.conv(tf.concat(inputs, 3))


class TFBottleneck(keras.layers.Layer):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, w=None):
        """
        Initialize a standard bottleneck layer for TensorFlow models, typically used for residual connections in a network.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            shortcut (bool, optional): Whether to include a shortcut connection. Default is True.
            g (int, optional): Number of groups for group convolution. Default is 1.
            e (float, optional): Expansion factor for hidden channels. Default is 0.5.
            w (object, optional): Pretrained weights from a PyTorch model to initialize the layer. Default is None.
        
        Returns:
            None
        
        Example:
            ```python
            import tensorflow as tf
        
            # Initialize the TFBottleneck layer
            c1, c2 = 64, 128
            bottleneck_layer = TFBottleneck(c1, c2)
        
            # Define input tensor
            inputs = tf.random.normal([1, 32, 32, c1])
        
            # Apply the bottleneck layer
            outputs = bottleneck_layer(inputs)
            print(outputs.shape)  # Expected output shape: (1, 32, 32, c2)
            ```
        
        Note:
            The bottleneck layer can be customized using pretrained weights for improved performance. Ensure the input tensor
            dimensions match the expected values when applying the bottleneck transformation.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_, c2, 3, 1, g=g, w=w.cv2)
        self.add = shortcut and c1 == c2

    def call(self, inputs):
        """
        Perform forward pass of the TFBottleneck module.
        
        Args:
            inputs (tf.Tensor): Input tensor with shape (N, H, W, C), where N is the batch size, H is the height,
                W is the width, and C is the number of channels.
        
        Returns:
            (tf.Tensor): Output tensor with shape (N, H, W, C2) where C2 is the number of output channels after the
                bottleneck operation.
        
        Example:
            ```python
            bottleneck = TFBottleneck(64, 128, shortcut=True)
            x = tf.random.uniform((1, 128, 128, 64))
            y = bottleneck(x)
            ```
        
        Note:
            If `self.add` is True, the function will add the input tensor to the convolution output. This typically occurs
            when the input and output channels are the same, and the `shortcut` parameter is set to True.
        """
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs))


class TFCrossConv(keras.layers.Layer):
    # Cross Convolution
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False, w=None):
        """
        Perform an enhanced cross convolution operation with optional expansion, grouping, and shortcut addition capabilities.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for the convolution operations. Default is 3.
            s (int): Stride size for the convolution operations. Default is 1.
            g (int): Number of groups for the grouped convolution. Default is 1.
            e (float): Expansion coefficient for intermediate channels. Default is 1.0.
            shortcut (bool): Whether to apply a shortcut connection (residual connection). Default is False.
            w (object | None): Pretrained weights object containing convolution parameters. Default is None.
        
        Returns:
            None: This constructor initializes an instance of the TFCrossConv class.
        
        Example:
            ```python
            cross_conv_layer = TFCrossConv(c1=32, c2=64, k=5, s=2, w=pretrained_weights)
            output = cross_conv_layer(inputs)  # Input tensor should have a shape compatible with these parameters
            ```
        
        Note:
            The cross convolution operation applies a two-step convolutions with different kernel shapes `(1, k)` and `(k, 1)`, 
            preceded by an optional expansion through an intermediate layer. When `shortcut` is True, the input is directly 
            added to the output of the two-step convolutions.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, (1, k), (1, s), w=w.cv1)
        self.cv2 = TFConv(c_, c2, (k, 1), (s, 1), g=g, w=w.cv2)
        self.add = shortcut and c1 == c2

    def call(self, inputs):
        """
        Perform cross convolution operations on input tensors.
        
        Args:
            inputs (tf.Tensor): Input tensor with shape (N, H, W, C), where N is the batch size, H is the height,
                W is the width, and C is the number of channels.
        
        Returns:
            (tf.Tensor): Tensor after applying cross convolution operations, with shape (N, H, W, C2) where C2 is the
                number of output channels.
        
        Example:
            ```python
            import tensorflow as tf
            
            # Define input tensor
            input_tensor = tf.random.normal([1, 64, 64, 32])
            
            # Initialize the TFCrossConv layer
            cross_conv_layer = TFCrossConv(c1=32, c2=64, k=3, s=1, g=1, e=1.0, shortcut=True)
            
            # Apply the layer
            output_tensor = cross_conv_layer(input_tensor)
            print(output_tensor.shape)  # Expected output shape: (1, 64, 64, 64)
            ```
        
        Note:
            If `shortcut` is True and the number of input channels (`c1`) equals the number of output channels (`c2`),
            a shortcut connection is added between the input and the output.
        """
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs))


class TFConv2d(keras.layers.Layer):
    # Substitution for PyTorch nn.Conv2D
    def __init__(self, c1, c2, k, s=1, g=1, bias=True, w=None):
        """
        Initialize a TensorFlow Conv2D layer, mimicking the behavior of PyTorch's Conv2D, with optional pretrained weights.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int | tuple[int, int]): Size of the convolutional kernel.
            s (int, optional): Stride size. Defaults to 1.
            g (int, optional): Number of groups. Only supported value is 1. Defaults to 1.
            bias (bool, optional): Whether to include a bias term. Defaults to True.
            w (torch.nn.Module | None, optional): Pretrained weights taken from a PyTorch model. Defaults to None.
        
        Example:
            ```python
            import torch
            from models.tf import TFConv2d
        
            # Define parameters
            c1, c2, k, s = 3, 64, 3, 1
        
            # Pretrained weights from a PyTorch model
            pretrained_weights = torch.nn.Conv2d(c1, c2, k)
        
            # Initialize TFConv2d layer
            conv_layer = TFConv2d(c1=c1, c2=c2, k=k, s=s, bias=True, w=pretrained_weights)
            ```
        
        Note:
            TensorFlow's `keras.layers.Conv2D` does not support the 'groups' argument prior to version 2.2, which limits
            `g` to 1.
        
        Returns:
            None: This constructor initializes the Conv2D layer within the class.
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
        Provide only the docstring content, without quotation marks or function code.
        
        Apply a convolution operation to the input tensor.
        
        Args:
            inputs (tf.Tensor): Input tensor with shape (B, H, W, C), where B is the batch size, H is the height,
                W is the width, and C is the number of channels.
        
        Returns:
            (tf.Tensor): Tensor resulting from the convolution operation, with shape (B, H_out, W_out, C_out) where H_out and
                W_out are the output height and width, and C_out is the number of output channels.
        
        Example:
            ```python
            import tensorflow as tf
            from models.tf import TFConv2d
        
            # Example input tensor
            input_tensor = tf.random.normal([1, 32, 32, 3])
        
            # Initialize Conv2D layer
            conv2d_layer = TFConv2d(c1=3, c2=16, k=3, s=1, bias=True, w=pretrained_weights)
        
            # Apply Conv2D layer
            output_tensor = conv2d_layer(input_tensor)
            ```
            
        Note:
            This function uses TensorFlow's Conv2D operation to simulate PyTorch's nn.Conv2D behavior. The layer supports only
            single-group convolutions (`g=1`) and padding is added manually when necessary.
        """
        return self.conv(inputs)


class TFBottleneckCSP(keras.layers.Layer):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        """
        Initializes CSP bottleneck layer with specified input/output channels, layer count, and network topology options.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of bottleneck layers. Default is 1.
            shortcut (bool): Whether to use shortcut connections or not.
            g (int): Number of groups for group convolution. Default is 1.
            e (float): Expansion ratio for hidden layers. Default is 0.5.
            w (object): Weights container to initialize the layers.
        
        Returns:
            (keras.layers.Layer): Constructed TensorFlow layer with CSP bottleneck configuration.
        
        Example:
            ```python
            csp_bottleneck = TFBottleneckCSP(c1=64, c2=128, n=2, shortcut=True, g=1, e=0.5, w=weights)
            output = csp_bottleneck(inputs)
            ```
        
        Note:
            Uses `TFConv` and `TFConv2d` for convolution operations, `TFBN` for batch normalization, and Keras Swish
            activation by default. This is based on the Cross Stage Partial Networks (CSPNet) architecture.
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
        Applies a CSP (Cross Stage Partial Networks) Bottleneck convolutional block to the input tensor.
        
        Args:
            inputs (tf.Tensor): Input tensor with shape (N, H, W, C), where N represents the batch size, H is the height, 
                W is the width, and C is the number of channels.
        
        Returns:
            (tf.Tensor): Output tensor after applying the CSP bottleneck block, which maintains the same batch size and 
                spatial dimensions but with modified channel dimensions depending on the convolutions' configurations.
        
        Example:
            ```python
            import tensorflow as tf
            from yolov5_models import TFBottleneckCSP
        
            # Define input tensor with shape (batch_size, height, width, channels)
            inputs = tf.random.normal([1, 128, 128, 64])
        
            # Initialize TFBottleneckCSP layer
            bottleneck_csp_layer = TFBottleneckCSP(c1=64, c2=128, n=1, shortcut=True, g=1, e=0.5)
        
            # Apply the layer to input tensor
            outputs = bottleneck_csp_layer(inputs)
            print(outputs.shape)  # Expected output shape: (1, 128, 128, 128)
            ```
        
        Note:
            The CSP architecture helps in strengthening gradient flow across the network, improving training dynamics, and 
            ensuring efficient parameter utilization.
        """
        y1 = self.cv3(self.m(self.cv1(inputs)))
        y2 = self.cv2(inputs)
        return self.cv4(self.act(self.bn(tf.concat((y1, y2), axis=3))))


class TFC3(keras.layers.Layer):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        """
        Perform CSP bottleneck operations with 3 convolutions, supporting optional shortcuts and group convolutions.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of bottleneck layers to apply.
            shortcut (bool): Determines whether to use shortcuts. Default is True.
            g (int): Number of groups for convolutions. Default is 1.
            e (float): Expansion ratio for bottleneck channels. Default is 0.5.
            w (torch.nn.Module | None): Pretrained weights from a PyTorch model to initialize the TensorFlow layer. Default is None.
        
        Returns:
            None: This method initializes the TFC3 layer.
        
        Example:
            ```python
            # Example usage of TFC3
            tfc3_layer = TFC3(c1=64, c2=128, n=3, shortcut=True, g=1, e=0.5)
            input_tensor = tf.random.normal([1, 64, 64, 64])
            output_tensor = tfc3_layer(input_tensor)
            ```
        
        Note:
            This layer implements CSPNet architecture with 3 convolutions, integrated in TensorFlow. Ideal for deep learning
            models that require efficient channel-wise transformations while maintaining the original network dimensions.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c1, c_, 1, 1, w=w.cv2)
        self.cv3 = TFConv(2 * c_, c2, 1, 1, w=w.cv3)
        self.m = keras.Sequential([TFBottleneck(c_, c_, shortcut, g, e=1.0, w=w.m[j]) for j in range(n)])

    def call(self, inputs):
        """
        Perform a forward pass through the CSP Bottleneck layer with 3 convolutions.
        
        Args:
            inputs (tf.Tensor): Input tensor to the layer, with shape (batch_size, height, width, channels).
        
        Returns:
            (tf.Tensor): Output tensor produced after applying CSP Bottleneck layer transformations.
        
        Example:
            ```python
            csp_bottleneck = TFC3(c1=64, c2=128, n=2, shortcut=True, g=1, e=0.5)
            output_tensor = csp_bottleneck(input_tensor)
            ```
        
        Note:
            This layer is part of the Ultralytics YOLOv5 model configuration for TensorFlow.
            See https://github.com/ultralytics/yolov5 for more details.
        """
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3))


class TFC3x(keras.layers.Layer):
    # 3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        """
        Perform non-maximum suppression (NMS) on prediction boxes.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int, optional): Number of CrossConv layers. Defaults to 1.
            shortcut (bool, optional): Whether to use shortcut connection. Defaults to True.
            g (int, optional): Number of groups for grouped convolution. Defaults to 1.
            e (float, optional): Expansion ratio. Defaults to 0.5.
            w (object, optional): Pretrained weights from a PyTorch model.
        
        Returns:
            None
        
        Note:
            This class is a part of TensorFlow, Keras, and TFLite versions of YOLOv5 as authored in
            https://github.com/ultralytics/yolov5/pull/1127. For usage, see https://github.com/ultralytics/yolov5.
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
        TFC3x.call(inputs)
        
        Processes input through cross-convolutions and merges features for enhanced detection.
        
        Args:
            inputs (tf.Tensor): Input tensor with shape (batch_size, height, width, channels).
        
        Returns:
            (tf.Tensor): Output tensor after processing through cross-convolutions and feature merging, with shape 
                (batch_size, new_height, new_width, new_channels).
        
        Example:
            ```python
            tfc3x_layer = TFC3x(c1=64, c2=128, n=3, shortcut=True, g=1, e=0.5)
            input_tensor = tf.random.normal([1, 64, 64, 64])
            output_tensor = tfc3x_layer(input_tensor)
            ```
            
        Note:
            This class is part of the TensorFlow, Keras, and TFLite versions of YOLOv5. See https://github.com/ultralytics/yolov5
            for more information.
        """
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3))


class TFSPP(keras.layers.Layer):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), w=None):
        """
        Initialize a spatial pyramid pooling (SPP) layer for YOLO models.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (tuple[int, int, int]): Kernel sizes for the spatial pooling layers. Default is (5, 9, 13).
            w (object | None): Weights from a pretrained model. Default is None.
        
        Returns:
            None
        
        Example:
            ```python
            yolo_spp = TFSPP(c1=256, c2=512, k=(5, 9, 13), w=pretrained_weights)
            ```
        
        Notes:
            The SPP layer is designed to increase the receptive field by applying a series of max pooling operations with 
            large kernel sizes, improving the detection of objects at various scales in YOLO models.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_ * (len(k) + 1), c2, 1, 1, w=w.cv2)
        self.m = [keras.layers.MaxPool2D(pool_size=x, strides=1, padding="SAME") for x in k]

    def call(self, inputs):
        """
        Perform spatial pyramid pooling (SPP) on the input tensor to extract multi-scale features.
        
        Args:
            inputs (tf.Tensor): Input tensor from the previous layer with shape (B, H, W, C), where B is the batch size, 
                H is the height, W is the width, and C is the number of input channels.
        
        Returns:
            (tf.Tensor): Output tensor with multi-scale features, after applying SPP and concatenation. The shape of 
                the output tensor will be (B, H, W, c2), where c2 is the number of output channels.
        
        Example:
            ```python
            spp_layer = TFSPP(c1=256, c2=512, k=(5, 9, 13))
            output = spp_layer(inputs)
            ```
        
        Note:
            The layer performs convolution and max pooling with different pool sizes before concatenating the results 
            for enhanced feature extraction. This is typically used in object detection models like YOLO for capturing 
            multi-scale context.
        """
        x = self.cv1(inputs)
        return self.cv2(tf.concat([x] + [m(x) for m in self.m], 3))


class TFSPPF(keras.layers.Layer):
    # Spatial pyramid pooling-Fast layer
    def __init__(self, c1, c2, k=5, w=None):
        """
        Initialize a TFSPPF (Spatial Pyramid Pooling-Fast) layer with specified parameters.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for max pooling. Default is 5.
            w (None | dict): Weights to initialize the layer. A dictionary containing the necessary weights for the layers.
        
        Returns:
            None: This method does not return anything as it initializes the layer in place.
        
        Example:
            ```python
            tf_sppf = TFSPPF(c1=256, c2=512, k=5, w=weights)
            ```
        
        Note:
            This TFSPPF layer is specifically designed for YOLOv5 architecture, offering a faster variant of spatial pyramid
            pooling by using fewer layers for efficiency while maintaining performance.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_ * 4, c2, 1, 1, w=w.cv2)
        self.m = keras.layers.MaxPool2D(pool_size=k, strides=1, padding="SAME")

    def call(self, inputs):
        """
        Perform spatial pyramid pooling-Fast (SPPF) on input tensors, concatenating pooled features with the original tensor.
        
        Args:
            inputs (tf.Tensor): Input tensor with shape (N, H, W, C) for batch size N, height H, width W, and channels C.
        
        Returns:
            (tf.Tensor): Output tensor with shape (N, H, W, C_out), where C_out is the number of output channels.
        
        Example:
            ```python
            layer = TFSPPF(c1=256, c2=512, k=5)
            output = layer(inputs)  # inputs should be a tensor of shape (N, H, W, 256)
            ```
        
        Note:
            This TFSPPF layer is specifically designed for YOLOv5 architecture, offering a faster variant of spatial pyramid
            pooling by using fewer layers for efficiency while maintaining performance.
        """
        x = self.cv1(inputs)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(tf.concat([x, y1, y2, self.m(y2)], 3))


class TFDetect(keras.layers.Layer):
    # TF YOLOv5 Detect layer
    def __init__(self, nc=80, anchors=(), ch=(), imgsz=(640, 640), w=None):
        """
        Initializes YOLOv5 detection layer for TensorFlow.
        
        Args:
            nc (int, optional): Number of classes. Defaults to 80.
            anchors (tuple, optional): Tuple of anchor box dimensions. Defaults to ().
            ch (tuple, optional): Number of input channels for each detection layer. Defaults to ().
            imgsz (tuple[int, int], optional): Input image size as (height, width). Defaults to (640, 640).
            w (object, optional): Weights object containing pretrained weight tensors and other parameters.
        
        Returns:
            None
        
        Note:
            This detection layer forms part of the YOLOv5 architecture for object detection tasks in TensorFlow, handling the
            prediction of bounding boxes and class probabilities for detected objects.
        
        Example:
            ```python
            detection_layer = TFDetect(nc=80, anchors=((10, 13, 16, 30, 33, 23),), ch=(256, 512, 1024), imgsz=(640, 640))
            ```
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
        Perform object detection computations using inputs from multiple feature layers, applying activation, convolution, and 
        grid-based adjustments to generate final output.
        
        Args:
            inputs (list[tf.Tensor]): List of input tensors from multiple feature layers, each with shape (B, H, W, C), 
                where B is batch size, H is height, W is width, and C is the number of channels.
        
        Returns:
            (tf.Tensor): Final processed tensor with object detection information, shape (B, N, 85), where N is the number 
                of predictions and 85 represents the output features (4 box coords + 1 objectness score + 80 class scores).
        
        Example:
            ```python
            detection_layer = TFDetect(nc=80, anchors=((10, 13, 16, 30, 33, 23),), ch=(256, 512, 1024), imgsz=(640, 640))
            output = detection_layer([feature_map1, feature_map2, feature_map3])
            ```
        
        Note:
            Ensure input tensors have consistent shapes aligned with the detection layer configuration. The function
            reshapes, normalizes, and concatenates features from each input tensor to produce final detection outputs.
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
        Generate a 2D coordinate grid for anchors with shape (1, 1, ny*nx, 2).
        
        Args:
            nx (int): Number of grid anchors along the x-axis. Default is 20.
            ny (int): Number of grid anchors along the y-axis. Default is 20.
        
        Returns:
            (tf.Tensor): A tensor containing the 2D coordinate grid with shape (1, 1, ny*nx, 2).
        
        Example:
            ```python
            grid = TFDetect._make_grid(20, 20)
            ```
        """
        # return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
        xv, yv = tf.meshgrid(tf.range(nx), tf.range(ny))
        return tf.cast(tf.reshape(tf.stack([xv, yv], 2), [1, 1, ny * nx, 2]), dtype=tf.float32)


class TFSegment(TFDetect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), imgsz=(640, 640), w=None):
        """
        Initialize the YOLOv5 segmentation head for TensorFlow models.
        
        Args:
            nc (int): Number of classes for segmentation.
            anchors (list[float]): List of anchor boxes used in YOLOv5, this should be an iterable containing anchor sizes.
            nm (int): Number of segmentation masks.
            npr (int): Number of prototypes.
            ch (list[int]): List of input channels for each detection layer.
            imgsz (tuple[int, int]): Image size in the format (height, width).
            w (object): Pretrained weights for initializing the model.
        
        Returns:
            None
        
        Example:
            ```python
            import tensorflow as tf
            from your_module import TFSegment
        
            segmentor = TFSegment(nc=80, anchors=[[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]],
                                  nm=32, npr=256, ch=[256, 512, 1024], imgsz=(640, 640), w=weights)
            ```
        
        Note:
            The 'w' parameter is critical for performance as it utilizes pretrained weights to enhance segmentation accuracy.
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
        Perform segmentation using the YOLOv5 segmentation head for TensorFlow models.
        
        Args:
            x (list[tf.Tensor]): Input feature maps from backbone network.
        
        Returns:
            (tuple[tf.Tensor]): A tuple containing:
                - detections (tf.Tensor): Detection predictions with shape (N, num_detections, 5 + num_classes + num_masks),
                  where N is the batch size.
                - proto (tf.Tensor): Prototype masks with shape (N, num_prototypes, height, width).
        
        Example:
            ```python
            # Assuming 'backbone_features' is a list of TensorFlow tensors from the backbone network
            segmentor = TFSegment(nc=80, anchors=[[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]],
                                  nm=32, npr=256, ch=[256, 512, 1024], imgsz=(640, 640), w=weights)
            detections, proto = segmentor(backbone_features)
            ```
        
        Note:
            The method processes the input feature maps to produce object detection predictions and prototype masks used in 
            segmentation tasks.
        """
        p = self.proto(x[0])
        # p = TFUpsample(None, scale_factor=4, mode='nearest')(self.proto(x[0]))  # (optional) full-size protos
        p = tf.transpose(p, [0, 3, 1, 2])  # from shape(1,160,160,32) to shape(1,32,160,160)
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p)


class TFProto(keras.layers.Layer):
    def __init__(self, c1, c_=256, c2=32, w=None):
        """
        Initializes TFProto layer with convolutional and upsampling layers for feature extraction and transformation.
        
        Args:
            c1 (int): Number of input channels.
            c_ (int): Number of hidden channels, default is 256.
            c2 (int): Number of output channels, default is 32.
            w (object | None): Pretrained weights for initializing the convolutional layers. If None, layers are initialized
                with default settings.
        
        Returns:
            (TFProto): Instance of the TFProto layer, ready for use in a TensorFlow model.
        
        Example:
            ```python
            tf_proto_layer = TFProto(c1=128)
            ```
        
        Note:
            This layer is designed to be a part of the YOLOv5 model pipeline, specifically for segmenting image features.
        """
        super().__init__()
        self.cv1 = TFConv(c1, c_, k=3, w=w.cv1)
        self.upsample = TFUpsample(None, scale_factor=2, mode="nearest")
        self.cv2 = TFConv(c_, c_, k=3, w=w.cv2)
        self.cv3 = TFConv(c_, c2, w=w.cv3)

    def call(self, inputs):
        """
        Handles forwarding through convolutional and upsampling layers to generate mask prototypes in TF models.
        
        Args:
            inputs (tf.Tensor): A tensor with shape (N, H, W, C), where N is the batch size, H is the height,
                W is the width, and C is the number of channels.
        
        Returns:
            (tf.Tensor): A tensor with the transformed features, having shape (N, H_new, W_new, C2) where H_new and W_new
                are the new height and width after processing, and C2 is the number of output channels.
        
        Example:
            ```python
            tf_proto_layer = TFProto(c1=128)
            input_tensor = tf.random.normal([1, 64, 64, 128])
            output_tensor = tf_proto_layer(input_tensor)
            ```
        """
        return self.cv3(self.cv2(self.upsample(self.cv1(inputs))))


class TFUpsample(keras.layers.Layer):
    # TF version of torch.nn.Upsample()
    def __init__(self, size, scale_factor, mode, w=None):
        """
        Initialize a TensorFlow upsampling layer.
        
        Args:
            size (tuple[int] | None): Desired output size. Default is `None`.
            scale_factor (int | None): Multiplier for the height and width of the input. Must be even. Default is `None`.
            mode (str): Upsampling algorithm to use. Options are ('nearest', 'bilinear', etc.).
            w (torch.nn.Module | None): Placeholder for compatibility. Default is `None`.
        
        Returns:
            None
        
        Example:
            ```python
            upsample_layer = TFUpsample(size=None, scale_factor=2, mode="nearest")
            result = upsample_layer(input_tensor)
            ```
        
        Note:
            Ensure that 'scale_factor' is a multiple of 2.
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
        Perform nearest neighbor upsampling on input tensors using the specified scale factor and mode in TensorFlow.
        
        Args:
            inputs (tf.Tensor): Input tensor to be upsampled, typically with shape (B, H, W, C) where B is the batch size,
                H is the height, W is the width, and C is the number of channels.
        
        Returns:
            (tf.Tensor): Upsampled tensor with dimensions equal to original dimensions multiplied by the scale factor.
                The output tensor will have a shape of (B, H * scale_factor, W * scale_factor, C).
        
        Example:
            ```python
            import tensorflow as tf
            from your_module import TFUpsample
        
            upsample_layer = TFUpsample(size=None, scale_factor=2, mode="nearest")
            input_tensor = tf.random.normal([1, 64, 64, 32])
            output_tensor = upsample_layer(input_tensor)
            print(output_tensor.shape)  # Expected output shape: (1, 128, 128, 32)
            ```
        
        Note:
            Ensure that 'scale_factor' is a multiple of 2. This layer resizes the spatial dimensions (height and width)
            of the input tensor by the specified scale factor.
        """
        return self.upsample(inputs)


class TFConcat(keras.layers.Layer):
    # TF version of torch.concat()
    def __init__(self, dimension=1, w=None):
        """
        Initializes a TensorFlow layer for concatenating tensors along the specified dimension.
        
        Args:
            dimension (int, optional): The dimension along which to concatenate tensors. Default is 1, converting 
                from NCHW to NHWC format.
            w (torch.nn.Module | None, optional): Pretrained weights from a PyTorch model to match the dimension 
                order. Default is None.
        
        Returns:
            None
        
        Example:
            ```python
            import tensorflow as tf
            from models.tf import TFConcat
        
            # Initialize the concatenate layer
            concat_layer = TFConcat(dimension=1)
        
            # Concatenate two sample tensors along the specified dimension
            tensor1 = tf.random.normal([1, 256, 256, 64])
            tensor2 = tf.random.normal([1, 256, 256, 64])
            output_tensor = concat_layer([tensor1, tensor2])
            ```
        
        Note:
            This class ensures compatibility between PyTorch and TensorFlow tensor formats by handling only the 
            concatenation of NCHW (PyTorch) to NHWC (TensorFlow).
        """
        super().__init__()
        assert dimension == 1, "convert only NCHW to NHWC concat"
        self.d = 3

    def call(self, inputs):
        """
        Concatenates input tensors along the last dimension, converting from NCHW to NHWC format.
        
        Args:
            inputs (list[tf.Tensor]): List of input tensors in NCHW format to be concatenated.
        
        Returns:
            (tf.Tensor): Concatenated tensor in NHWC format.
        
        Example:
            ```python
            concat_layer = TFConcat()
            input1 = tf.random.normal([1, 64, 32, 32])
            input2 = tf.random.normal([1, 64, 32, 32])
            output = concat_layer([input1, input2])
            ```
        """
        return tf.concat(inputs, self.d)


def parse_model(d, ch, model, imgsz):
    """
    Parses the model configuration dictionary to create YOLOv5 model layers with dynamic channel adjustments.
    
    This function processes the model configuration, initializing layers and setting up the neural network architecture
    for YOLOv5 model training and inference in TensorFlow.
    
    Args:
        d (dict): Model configuration dictionary containing anchor boxes, class counts, depth multiple, width multiple, 
            and backbone/head layer details.
        ch (list[int]): List of input channels for each layer.
        model (object): Existing model object containing pretrained weights and other parameters.
        imgsz (tuple[int, int]): Input image size as (height, width).
    
    Returns:
        (list[keras.Sequential]): Parsed list of keras.Sequential model layers set up for YOLOv5, with adjusted channels.
    
    Example:
        ```python
        model_config = {
            'anchors': [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]],
            'nc': 80, 'depth_multiple': 0.33, 'width_multiple': 0.50,
            'backbone': [
                [-1, 1, 'Conv', [32, 3, 1]],
                [-1, 1, 'C3', [64, 3, 0.5]],
            ],
            'head': [
                [-1, 1, 'SPPF', [256, 5]],
            ],
        }
        input_channels = [3]
        parsed_model_layers = parse_model(model_config, input_channels, pretrained_model, (640, 640))
        ```
    
    Note:
        This function converts PyTorch model layers to TensorFlow Keras layers, maintaining parameter consistency.
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
        Initialize a TensorFlow YOLOv5 model with specified configuration, input channels, and classes.
        
        Args:
            cfg (str | dict): Model configuration, either a file path to a yaml file or a dictionary containing network
                structure and parameters.
            ch (int): Number of input channels.
            nc (int | None): Number of classes for detection tasks.
            model (torch.nn.Module | None): PyTorch model instance to map to TensorFlow model structure.
            imgsz (tuple[int, int]): Input image size as a tuple of (height, width).
        
        Returns:
            None
        
        Example:
            ```python
            tf_model = TFModel(cfg='yolov5s.yaml', ch=3, nc=80, imgsz=(640, 640))
            ```
        
        Note:
            This model supports YOLOv5 architectures and is compatible with TensorFlow and Keras frameworks.
            Ensure to provide properly formatted configuration files or dictionaries for successful model initialization.
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
        Perform prediction on input data using the TensorFlow YOLOv5 model, optionally applying non-max suppression.
        
        Args:
            inputs (tf.Tensor): Input tensor containing the image data, shape (B, H, W, C).
            tf_nms (bool): Apply TensorFlow non-max suppression after prediction. Default is False.
            agnostic_nms (bool): Class-agnostic non-max suppression. Default is False.
            topk_per_class (int): Top-K maximum detections per class. Default is 100.
            topk_all (int): Top-K maximum total detections. Default is 100.
            iou_thres (float): Intersection-over-union (IoU) threshold for NMS. Default is 0.45.
            conf_thres (float): Confidence score threshold for filtering predictions. Default is 0.25.
        
        Returns:
            (tuple[tf.Tensor]): Tuple containing predicted bounding boxes, confidence scores, and class probabilities. 
                If `tf_nms` is True, returns the results of TensorFlow NMS, shape (N, 7) where N is the number of predictions 
                with columns for (x1, y1, x2, y2, score, class). If `tf_nms` is False, returns the raw tensor output from the model.
        
        Example:
            ```python
            import tensorflow as tf
            from ultralytics import TFModel
        
            # Initialize the model
            model = TFModel(cfg='yolov5s.yaml', ch=3, nc=80)
        
            # Prepare input tensor
            img = tf.random.normal([1, 640, 640, 3])
        
            # Perform prediction
            predictions = model.predict(img, tf_nms=True)
            ```
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
        Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2].
        
        Args:
            xywh (torch.Tensor): Bounding boxes in the format (x, y, w, h) with shape (N, 4) where N is the number of boxes.
        
        Returns:
            (torch.Tensor): Bounding boxes in the format (x1, y1, x2, y2) with shape (N, 4), where x1, y1 are top-left coordinates,
            and x2, y2 are bottom-right coordinates.
        
        Notes:
            This method is useful for converting bounding box formats for various operations like plotting, Non-Maximum 
            Suppression (NMS), or further model predictions.
        
        Examples:
            ```python
            boxes_xywh = torch.Tensor([[50, 50, 100, 100], [30, 40, 120, 80]])
            boxes_xyxy = TFModel._xywh2xyxy(boxes_xywh)
            ```
        """
        x, y, w, h = tf.split(xywh, num_or_size_splits=4, axis=-1)
        return tf.concat([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)


class AgnosticNMS(keras.layers.Layer):
    # TF Agnostic NMS
    def call(self, input, topk_all, iou_thres, conf_thres):
        """
        Perform class-agnostic non-maximum suppression (NMS) on input bounding boxes.
        
        Args:
            input (tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Tuple containing:
                boxes (tf.Tensor): Bounding boxes with shape (N, 4), where N is the number of boxes.
                classes (tf.Tensor): Class predictions with shape (N,), where N is the number of boxes.
                scores (tf.Tensor): Confidence scores with shape (N, C), where C is the number of classes.
            topk_all (int): Maximum number of final boxes to keep after non-max suppression.
            iou_thres (float): Intersection over union (IoU) threshold for NMS.
            conf_thres (float): Confidence threshold to filter boxes before NMS.
        
        Returns:
            (tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]): Four tensors containing:
                boxes (tf.Tensor): Filtered bounding boxes after NMS, with shape (M, 4), where M is the number of kept boxes.
                scores (tf.Tensor): Scores of kept boxes, with shape (M,).
                classes (tf.Tensor): Class indices of kept boxes, with shape (M,).
                indices (tf.Tensor): Original indices of the kept boxes, with shape (M,).
        
        Example:
            ```python
            boxes = tf.random.uniform((100, 4), minval=0, maxval=640)
            scores = tf.random.uniform((100, 80), minval=0, maxval=1)
            classes = tf.argmax(scores, axis=1)
            input = (boxes, classes, scores)
            nms_layer = AgnosticNMS()
            final_boxes, final_scores, final_classes, final_indices = nms_layer(
                input, topk_all=20, iou_thres=0.5, conf_thres=0.25)
            ```
        
        Note:
            The function is designed to operate on single-anchor format inputs, performing class-agnostic NMS to reduce redundancy 
            in detected bounding boxes.
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
        Perform agnostic non-maximum suppression on given bounding box predictions.
        
        Args:
            input (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing bounding boxes (N, 4), classes (N, C) 
                where C is the number of classes, and scores (N, C) where N is the number of predictions.
            topk_all (int): The maximum number of detections to keep.
            iou_thres (float): Intersection over Union (IoU) threshold for NMS.
            conf_thres (float): Confidence threshold for filtering low-confidence predictions.
        
        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Padded bounding boxes with shape (topk_all, 4).
                - torch.Tensor: Padded scores with shape (topk_all).
                - torch.Tensor: Padded class indices with shape (topk_all).
                - int: Number of valid detections.
        
        Example:
            ```python
            boxes = torch.rand(100, 4)
            classes = torch.rand(100, 5)
            scores = torch.rand(100, 5)
            topk_all = 50
            iou_thres = 0.5
            conf_thres = 0.3
        
            selected_boxes, padded_scores, selected_classes, valid_detections = AgnosticNMS._nms(
                (boxes, classes, scores), topk_all, iou_thres, conf_thres)
            ```
        
        Note:
            This function considers detections class-agnostic and clusters all predicted boxes without regard to class labels 
            during NMS.
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
    Convert PyTorch activation functions to their TensorFlow equivalents.
    
    Args:
        act (type[torch.nn.Module], optional): Activation function from PyTorch. Default is nn.SiLU.
    
    Returns:
        (callable): A TensorFlow-compatible activation function.
    
    Example:
        ```python
        tf_activation = activations(nn.LeakyReLU)
        output = tf_activation(input_tensor)
        ```
    
    Note:
        Supports the conversion of LeakyReLU, Hardswish, and SiLU (Swish) activation functions. For unsupported types, 
        raises an error.
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
    Generates a representative dataset for calibration by yielding transformed numpy arrays from the input dataset.
    
    Args:
        dataset (iterable): Dataset to yield images for calibration. Each item in the dataset should be a tuple containing
            (path, img, im0s, vid_cap, string), where 'img' is the image represented as a numpy array with shape (C, H, W).
        ncalib (int): Number of samples to yield for calibration (default is 100).
    
    Returns:
        (generator): A generator yielding a list of numpy arrays, each representing an image with shape (1, H, W, C) scaled and
            preprocessed for model calibration.
    
    Example:
        ```python
        dataset = DataLoader(...)  # define your dataset
        data_gen = representative_dataset_gen(dataset, ncalib=50)
        for calibration_data in data_gen:
            # perform calibration
        ```
    
    Notes:
        - The function stops yielding data once ncalib samples have been produced from the dataset.
        - Images are converted from shape (C, H, W) to (1, H, W, C) and scaled to a range of [0, 1].
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
    """
    Exports YOLOv5 model from PyTorch to TensorFlow and Keras formats, performing inference for validation.
    
    Args:
        weights (str | pathlib.Path): Path to the weights file. Default is ROOT / "yolov5s.pt".
        imgsz (tuple[int, int]): Tuple of integers representing the height and width of the image for inference. Default is (640, 640).
        batch_size (int): Size of the batch for inference. Default is 1.
        dynamic (bool): Flag to indicate if dynamic batch size should be used in Keras model. Default is False.
    
    Returns:
        None: The function exports the model and performs inference without returning any value.
    
    Example:
        ```python
        run(weights='best.pt', imgsz=(640, 640), batch_size=1, dynamic=False)
        ```
    
    Note:
        - Ensure you have the necessary dependencies installed (`torch`, `tensorflow`, `keras`).
        - Adjust the `weights` path, `imgsz`, `batch_size`, and `dynamic` flag as needed for your setup.
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
    Parse command-line arguments for model inference configuration.
    
    This utility function parses command-line arguments to configure the inference properties such as paths to weight files,
    image sizes, batch sizes, and dynamic batch size options.
    
    Args:
        None
    
    Returns:
        (argparse.Namespace): Namespace object containing parsed command-line options:
            - weights (str): Path to the model weights.
            - imgsz (list[int]): Inference image size (height, width).
            - batch_size (int): Batch size for inference.
            - dynamic (bool): Whether to use dynamic batch size.
    
    Example:
        ```python
        opt = parse_opt()
        print(opt.weights)
        print(opt.imgsz)
        print(opt.batch_size)
        print(opt.dynamic)
        ```
    
    Note:
        The --imgsz argument accepts either a single integer or a tuple of two integers. If only one value is provided,
        it will be duplicated to form a square shape (height, width).
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
    Execute the main function to run model export and validation processes for YOLOv5, including conversion to TensorFlow 
    and Keras formats.
    
    Args:
        opt (argparse.Namespace): Parsed command-line arguments which include:
            - weights (str): Path to the model weights.
            - imgsz (list[int]): Inference image size (height, width).
            - batch_size (int): Batch size for inference.
            - dynamic (bool): Whether to use dynamic batch size.
    
    Example:
        ```python
        if __name__ == "__main__":
            opt = parse_opt()
            main(opt)
        ```
    
    Note:
        This function integrates and validates the conversion of YOLOv5 from PyTorch to TensorFlow and Keras frameworks.
        For additional export options, refer to the export.py script.
    """
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
