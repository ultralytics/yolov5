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
        Initializes a TensorFlow BatchNormalization layer with optional pretrained weights.

        Args:
            w (torch.nn.BatchNorm2d, optional): A PyTorch BatchNorm2d layer whose weights and biases are used to initialize the TensorFlow BatchNormalization layer.

        Returns:
            None

        Notes:
            This class is designed as a wrapper to leverage pre-trained PyTorch weights for batch normalization in TensorFlow, enabling seamless integration of TensorFlow within the Ultralytics YOLOv5 framework.

        Example:
            ```python
            # Using TFBN with a pre-trained PyTorch BatchNorm2d layer
            import torch
            from tensorflow.keras.layers import Input

            pytorch_bn = torch.nn.BatchNorm2d(num_features=128)
            tf_bn_layer = TFBN(w=pytorch_bn)

            input_tensor = Input(shape=(128, 128, 128))
            output_tensor = tf_bn_layer(input_tensor)
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
        Applies batch normalization to the inputs.

        Args:
            inputs (tf.Tensor): Input tensor to be batch normalized. Typically, a 4D tensor with shape
                (batch_size, height, width, channels).

        Returns:
            tf.Tensor: Output tensor with the same shape as `inputs`, after batch normalization is applied.

        Notes:
            For more information on Keras BatchNormalization, refer to the Keras documentation:
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalizati
        """
        return self.bn(inputs)


class TFPad(keras.layers.Layer):
    # Pad inputs in spatial dimensions 1 and 2
    def __init__(self, pad):
        """
        Initializes a padding layer for spatial dimensions 1 and 2 with specified padding, supporting both int and tuple
        inputs.

        Args:
            pad (int | tuple): Padding value. If an integer, the same padding is applied to all sides.
                If a tuple, it should contain two elements (pad_height, pad_width) specifying the padding for the height
                and width dimensions, respectively.

        Returns:
            None: This constructor initializes the TFPad layer with the specified padding configuration.

        Notes:
            The padding applied is in the format `[[0, 0], [pad_height, pad_height], [pad_width, pad_width], [0, 0]]`,
            which corresponds to the batch dimension (no padding), height dimension, width dimension, and channel dimension (no padding).

        Examples:
        ```python
        # Applying symmetric padding of 2 to height and width dimensions
        layer = TFPad(2)

        # Applying different padding for height (2) and width (3)
        layer = TFPad((2, 3))
        ```
        """
        super().__init__()
        if isinstance(pad, int):
            self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
        else:  # tuple/list
            self.pad = tf.constant([[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]])

    def call(self, inputs):
        """
        Pads input tensor with zeros in spatial dimensions according to specified padding configuration.

        Args:
            inputs (tf.Tensor): Input tensor to be padded.

        Returns:
            tf.Tensor: Padded tensor with zeros as specified by the padding configuration.

        Examples:
            ```python
            import tensorflow as tf
            from ultralytics import TFPad

            # Int padding
            pad_layer = TFPad(2)
            input_tensor = tf.random.normal([1, 28, 28, 3])
            padded_tensor = pad_layer(input_tensor)

            # Tuple padding
            pad_layer = TFPad((2, 3))
            input_tensor = tf.random.normal([1, 28, 28, 3])
            padded_tensor = pad_layer(input_tensor)
            ```
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
            k (int | tuple[int, int], optional): Kernel size. Default is 1.
            s (int | tuple[int, int], optional): Stride size. Default is 1.
            p (int | tuple[int, int] | None, optional): Padding size. None uses 'SAME' padding if stride is 1, and 'VALID'
                padding otherwise. Default is None.
            g (int, optional): Number of groups for group convolution. Must be 1 for TensorFlow version. Default is 1.
            act (bool, optional): Whether to include an activation function. Default is True.
            w (torch.nn.Module, optional): PyTorch module containing the weights for Conv2D and BatchNormalization. Default is None.

        Returns:
            None: This is an initializer method that sets up layers for the TensorFlow convolutional layer with optional batch
                normalization and activation function.
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
        Applies convolution, batch normalization, and activation function to input tensors.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, height, width, channels) to be processed.

        Returns:
            tf.Tensor: Output tensor after applying convolution, batch normalization, and activation function.
        """
        return self.act(self.bn(self.conv(inputs)))


class TFDWConv(keras.layers.Layer):
    # Depthwise convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True, w=None):
        """
        Initializes a depthwise convolution layer for TensorFlow models, with optional batch normalization and
        activation function, suitable for use in YOLOv5.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels. Must be a multiple of c1.
            k (int, optional): Kernel size for the depthwise convolution. Default is 1.
            s (int, optional): Stride of the convolution. Default is 1.
            p (int | None, optional): Padding value. Default is None.
            act (bool, optional): Whether to use an activation function. Default is True.
            w (torch.nn.Module | None, optional): Pretrained weights for initializing the convolution. Default is None.

        Returns:
            None

        Notes:
            TensorFlow convolution padding is inconsistent with PyTorch. The 'VALID' padding is used when the stride is not 1,
            otherwise 'SAME' padding is applied. This ensures compatibility with YOLOv5 model architecture requirements.

        Examples:
            ```python
            import tensorflow as tf
            from your_module import TFDWConv

            # Example initialization
            layer = TFDWConv(32, 64, k=3, s=2, p=1, act=True, w=pretrained_weights)
            output = layer(input_tensor)
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
        Applies depthwise convolution, batch normalization, and activation function to input tensors.

        Args:
            inputs (tf.Tensor): Input tensor to apply depthwise convolution.

        Returns:
            tf.Tensor: The processed tensor after applying depthwise convolution, optional batch normalization,
            and activation function.

        Notes:
            - This function is primarily designed to be used in TensorFlow models within the Ultralytics framework.
            - Ensure the input tensor matches the expected input dimensions for the initialized depthwise convolution layer.
        """
        return self.act(self.bn(self.conv(inputs)))


class TFDWConvTranspose2d(keras.layers.Layer):
    # Depthwise ConvTranspose2d
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0, w=None):
        """
        Initializes a Depthwise ConvTranspose2D layer with specific channel, kernel, stride, and padding settings.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels. Must be equal to c1.
            k (int): Kernel size. Must be 4.
            s (int): Stride size.
            p1 (int): Padding, first dimension. Must be 1.
            p2 (int): Padding, second dimension.
            w (torch.nn.Parameter): Pretrained weights for initializing the kernel and bias.

        Returns:
            None

        Notes:
            This layer only supports configurations where the kernel size `k` is 4 and the padding `p1` is 1. It asserts
            that `c1` and `c2` are equal, ensuring that the number of input and output channels are the same. The Conv2DTranspose
            layers are initialized separately for each channel.

        Examples:
            ```python
            from models.common import TFDWConvTranspose2d

            # Example initialization
            depthwise_transpose_layer = TFDWConvTranspose2d(c1=64, c2=64, k=4, s=2, p1=1, p2=1, w=your_pretrained_weights)
            ```
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
        Processes input through depthwise ConvTranspose2d layers and concatenates results after trimming border pixels.

        Args:
            inputs (tf.Tensor): Input tensor to be processed by the depthwise ConvTranspose2d layers.

        Returns:
            tf.Tensor: Output tensor after applying the depthwise ConvTranspose2d layers and concatenating the results.
        """
        return tf.concat([m(x) for m, x in zip(self.conv, tf.split(inputs, self.c1, 3))], 3)[:, 1:-1, 1:-1]


class TFFocus(keras.layers.Layer):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):
        """
        Initializes the TFFocus layer to aggregate spatial (width and height) information into the channel dimension
        using convolution.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for convolution. Defaults to 1.
            s (int): Stride for convolution. Defaults to 1.
            p (int | None): Padding for convolution. Defaults to None.
            g (int): Number of groups for convolution. Defaults to 1.
            act (bool): Whether to use an activation function. Defaults to True.
            w (torch.nn.Module | None): PyTorch weights to initialize the TensorFlow layers. Defaults to None.

        Returns:
            None

        Notes:
            This layer is crucial in YOLOv5 for spatial information aggregation, contributing to the network's ability to
            encode positional information effectively.

        Example:
            ```python
            focus_layer = TFFocus(c1=32, c2=64, k=3, s=1)
            output = focus_layer(input_tensor)
            ```
        """
        super().__init__()
        self.conv = TFConv(c1 * 4, c2, k, s, p, g, act, w.conv)

    def call(self, inputs):
        """
        Performs pixel shuffling and convolution on the input tensor, downsampling by a factor of 2 and expanding the
        number of channels by a factor of 4.

        Args:
            inputs (tf.Tensor): Input tensor with shape (batch, height, width, channels).

        Returns:
            tf.Tensor: Tensor with shape (batch, height/2, width/2, 4*channels) after pixel shuffling and convolution.
        """
        inputs = [inputs[:, ::2, ::2, :], inputs[:, 1::2, ::2, :], inputs[:, ::2, 1::2, :], inputs[:, 1::2, 1::2, :]]
        return self.conv(tf.concat(inputs, 3))


class TFBottleneck(keras.layers.Layer):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, w=None):
        """
        Initializes a standard bottleneck layer for TensorFlow models, expanding and contracting channels with optional
        shortcut.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            shortcut (bool): If True, adds a residual shortcut connection. Default is True.
            g (int): Number of groups for grouped convolution. Default is 1.
            e (float): Expansion factor for hidden layer channels. Default is 0.5.
            w (object): Weights for the layer, used to initialize the convolution layers.

        Returns:
            None

        Examples:
            ```python
            bottleneck = TFBottleneck(64, 128, shortcut=True, g=1, e=0.5, w=pretrained_weights)
            output = bottleneck(input_tensor)
            ```
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_, c2, 3, 1, g=g, w=w.cv2)
        self.add = shortcut and c1 == c2

    def call(self, inputs):
        """
        A comprehensive Google-style docstring for the `TFBottleneck.call` method is as follows:

        Performs forward pass; if shortcut is True & input/output channels match, adds input to the convolution result.

        Args:
            inputs (tf.Tensor): Input tensor with shape (batch_size, height, width, channels).

        Returns:
            tf.Tensor: Output tensor after applying bottleneck block operations, preserving input dimensions if shortcut is enabled.

        Notes:
            The bottleneck layer includes two convolutional layers. The first reduces the channel dimensions while the second restores them. If shortcut is enabled (and input/output channels are the same), the input will be added to the output of the convolutions.

        Examples:
            ```python
            import tensorflow as tf
            from some_module import TFBottleneck

            # Initialize the bottleneck layer
            bottleneck = TFBottleneck(c1=64, c2=128, shortcut=True)

            # Create a dummy input tensor
            x = tf.random.normal((1, 32, 32, 64))

            # Perform a forward pass
            output = bottleneck(x)
            print(output.shape)  # Should be (1, 32, 32, 128)
            ```
        """
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs))


class TFCrossConv(keras.layers.Layer):
    # Cross Convolution
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False, w=None):
        """
        Initializes a cross convolution layer with optional expansion, grouping, and shortcut addition capabilities.

        Args:
          c1 (int): Number of input channels.
          c2 (int): Number of output channels.
          k (int | tuple): Kernel size (default is 3).
          s (int): Stride (default is 1).
          g (int): Number of groups for grouped convolution (default is 1).
          e (float): Expansion ratio to control hidden channels (default is 1.0).
          shortcut (bool): Whether to use a shortcut connection (default is False).
          w (torch.nn.Module | None): Optional pretrained weights (default is None).

        Returns:
          None

        Notes:
          Designed specifically for TensorFlow implementations of YOLOv5 models. This layer performs two convolutions with
          different kernel shapes and combines their output, optionally utilizing grouping and shortcut connections to
          enhance learning capacity and performance.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, (1, k), (1, s), w=w.cv1)
        self.cv2 = TFConv(c_, c2, (k, 1), (s, 1), g=g, w=w.cv2)
        self.add = shortcut and c1 == c2

    def call(self, inputs):
        """
        Initializes a cross convolution layer comprising two convolutions and an optional shortcut connection.

        Args:
          c1 (int): The number of input channels.
          c2 (int): The number of output channels.
          k (int, optional): The size of the convolution kernel. Defaults to 3.
          s (int, optional): The stride of the convolution. Defaults to 1.
          g (int, optional): The number of groups for grouped convolution. Defaults to 1.
          e (float, optional): Expansion factor for intermediate channels. Defaults to 1.0.
          shortcut (bool, optional): Whether to add a shortcut connection if input and output channels match. Defaults to False.
          w (object, optional): Pre-trained weights for the convolution layers. Defaults to None.

        Returns:
          tf.Tensor: The output tensor after applying cross convolution and optional shortcut connection.

        Notes:
          - Conducts successive convolutions with asymmetric kernels (1xk followed by kx1).
          - Supports grouping only in the second convolution.
          - Shortcut connection is added only if the input and output channels are the same.

        Example:
        ```python
        import tensorflow as tf
        from models.tf import TFCrossConv

        # Initialize TFCrossConv
        cross_conv = TFCrossConv(c1=32, c2=64, k=3, s=1, shortcut=True)

        # Call TFCrossConv with input tensor
        input_tensor = tf.random.normal([8, 128, 128, 32])
        output_tensor = cross_conv(input_tensor)
        ```

        References:
        - https://github.com/ultralytics/yolov5/pull/1127
        - https://github.com/zldrobit

        Experimental usage:
        ```bash
        $ python models/tf.py --weights yolov5s.pt
        $ python export.py --weights yolov5s.pt --include saved_model pb tflite tfjs
        ```
        """
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs))


class TFConv2d(keras.layers.Layer):
    # Substitution for PyTorch nn.Conv2D
    def __init__(self, c1, c2, k, s=1, g=1, bias=True, w=None):
        """
        Initializes a TensorFlow 2D convolution layer, replicating PyTorch's nn.Conv2D functionality with support for
        configurable filter sizes, strides, and biases.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size, specifying the dimensions of the convolution filter.
            s (int, optional): Stride size for the convolution operation. Defaults to 1.
            g (int, optional): Number of groups for Grouped Convolutions. Defaults to 1. Note: This feature is
                not supported and must be set to 1.
            bias (bool, optional): Boolean indicator to include bias parameters. Defaults to True.
            w (torch.nn.Conv2d, optional): PyTorch Conv2D layer weights for initialization.

        Raises:
            AssertionError: If 'groups' (g) argument is not equal to 1, as group convolutions are not supported in this version
                of TensorFlow Conv2D.

        Returns:
            None

        Notes:
            TensorFlow's Conv2D padding behaves differently from PyTorch's Conv2D padding. Explicit padding
            might be necessary in certain cases to maintain consistent behavior across frameworks.

        Example:
            ```python
            # Initializing TFConv2d
            torch_layer = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
            tf_conv2d_layer = TFConv2d(c1=3, c2=16, k=3, s=1, bias=True, w=torch_layer)
            ```
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
        Performs a 2D convolution operation on the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor with shape (batch, height, width, channels), representing the data to be convolved.

        Returns:
            tf.Tensor: Output tensor resulting from the convolution operation, includes the shape transformation according to the
            specified filters, kernel size, and stride.

        Note:
            This function is designed to mimic the behavior of PyTorch's nn.Conv2D within TensorFlow, including initializing
            weights from pretrained models.
        """
        return self.conv(inputs)


class TFBottleneckCSP(keras.layers.Layer):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        """
        Initializes CSP bottleneck layer with specified channel configurations, number of bottlenecks, shortcut options,
        groups, and expansion factor.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of bottleneck layers.
            shortcut (bool): If True, adds shortcut connections inside the bottleneck.
            g (int): Number of groups for grouped convolution.
            e (float): Expansion factor to determine hidden channel size.
            w (object): Weights object containing pretrained weights.

        Returns:
            None

        Notes:
            CSP stands for Cross Stage Partial Network. For more details, visit the original paper at
            https://github.com/WongKinYiu/CrossStagePartialNetworks.

        Examples:
            ```python
            bottleneck_csp = TFBottleneckCSP(c1=64, c2=128, n=3, shortcut=True, g=1, e=0.5, w=weights)
            ```
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
        """TFBottleneckCSP.call(self, inputs):"""
            Processes input through the CSP bottleneck model layers, concatenates intermediate results, applies batch 
            normalization, and activation to produce the final output.
        
            Args:
                inputs (tf.Tensor): The input tensor to be processed, usually of shape (batch_size, height, width, channels).
        
            Returns:
                tf.Tensor: Output tensor after processing through the CSP bottleneck layers, typically with the same spatial 
                resolution but possibly different channel depth due to convolution operations.
        """
        y1 = self.cv3(self.m(self.cv1(inputs)))
        y2 = self.cv2(inputs)
        return self.cv4(self.act(self.bn(tf.concat((y1, y2), axis=3))))


class TFC3(keras.layers.Layer):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        """
        TFC3(c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None)
        
        Initializes a CSP (Cross Stage Partial) Bottleneck layer for TensorFlow models with 3 convolutions,
        supporting optional shortcuts and group convolutions.
        
        Args:
          c1 (int): Number of input channels.
          c2 (int): Number of output channels.
          n (int): Number of bottleneck layers. Defaults to 1.
          shortcut (bool): Use shortcuts. Defaults to True.
          g (int): Number of groups for group convolutions. Defaults to 1.
          e (float): Expansion ratio for the bottleneck. Defaults to 0.5.
          w (any): Weights for initializing the layer. Defaults to None.
        
        Returns:
          None
        
        Example:
          ```python
          layer = TFC3(64, 128, n=3, shortcut=False, g=2, e=0.5)
          output = layer(input_tensor)
          ```
        
        Note:
          This layer is specifically designed for compatibility with YOLOv5 models exported to TensorFlow,
          Keras, and TFLite formats.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c1, c_, 1, 1, w=w.cv2)
        self.cv3 = TFConv(2 * c_, c2, 1, 1, w=w.cv3)
        self.m = keras.Sequential([TFBottleneck(c_, c_, shortcut, g, e=1.0, w=w.m[j]) for j in range(n)])

    def call(self, inputs):
        """
        TFC3.call(inputs):
            """
            Forward pass for the CSP Bottleneck with 3 convolutions, designed to enhance feature extraction by merging
            two parallel transformations.

            Args:
                inputs (tf.Tensor): A 4D tensor of shape (batch_size, height, width, channels) representing the input feature
                maps.

            Returns:
                tf.Tensor: A 4D tensor of shape (batch_size, new_height, new_width, new_channels) after applying the CSP
                Bottleneck transformations. The dimensions are determined by the initialized layer parameters.

            Notes:
                - The layer applies two parallel convolutions followed by a concatenation and a further convolution to
                  combine the features.
                - Shortcut connections can be optionally configured for deep network architectures.

            Example:
                ```python
                import tensorflow as tf
                from models.tf import TFC3

                # Initialize layer
                csp_bottleneck = TFC3(c1=64, c2=128, n=3, shortcut=True)
                input_tensor = tf.random.normal([1, 128, 128, 64])  # Example input tensor
                output_tensor = csp_bottleneck(input_tensor)
                print(output_tensor.shape)  # Expected output shape: (1, 128, 128, 128)
                ```
            """
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3))


class TFC3x(keras.layers.Layer):
    # 3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        """
        Initializes TFC3x layer designed for YOLOv5 architecture, integrating cross-convolutions for object detection
        tasks.

        Args:
          c1 (int): Number of input channels.
          c2 (int): Number of output channels.
          n (int): Number of sequential layers to apply within the module.
          shortcut (bool): Whether to include shortcuts in the module, enhancing gradient flow.
          g (int): Number of groups for grouped convolutions.
          e (float): Expansion rate controlling the number of hidden channels.
          w (dict): Pretrained weights for initialization of the layers, expected keys include 'cv1', 'cv2', 'cv3', and 'm'.

        Returns:
          None

        Notes:
          This class plays a critical role in the YOLOv5 architecture for improving feature extraction.
          See https://github.com/ultralytics/yolov5 for further details on the usage and context of this layer.

        Examples:
          ```python
          import tensorflow as tf
          w = {...}  # Assume weights loaded for the model
          layer = TFC3x(c1=32, c2=64, n=1, shortcut=True, g=1, e=0.5, w=w)
          inputs = tf.random.normal([1, 224, 224, 32])  # Batch size of 1, 224x224 image, 32 channels
          outputs = layer(inputs)
          ```
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
        Processes input through cascaded convolutions and merges features, returning the final tensor output.

        Args:
            inputs (tf.Tensor): Input tensor to the layer.

        Returns:
            tf.Tensor: Processed tensor output after applying cross-convolutions and other operations.
        """
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3))


class TFSPP(keras.layers.Layer):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), w=None):
        """
        Initializes the TensorFlow Spatial Pyramid Pooling (SPP) layer commonly used in YOLOv3-SPP models for enhanced
        receptive fields.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (tuple[int]): Tuple of kernel sizes for the pooling layers, defining the receptive field sizes. Default is (5, 9, 13).
            w (Any): Weights of the layer, typically from a pretrained model.

        Returns:
            None

        Notes:
            The TFSPP layer helps in aggregating multi-scale contextual information by using multiple max-pooling operations
            with varying kernel sizes. This aggregation aids in object detection tasks by capturing features at different
            scales.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_ * (len(k) + 1), c2, 1, 1, w=w.cv2)
        self.m = [keras.layers.MaxPool2D(pool_size=x, strides=1, padding="SAME") for x in k]

    def call(self, inputs):
        """
        Processes input tensor through the YOLOv3-SPP layer, applying max pooling with different kernel sizes and
        concatenating the results.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, height, width, channels).

        Returns:
            tf.Tensor: Output tensor after applying spatial pyramid pooling and convolutional transformations.
        """
        x = self.cv1(inputs)
        return self.cv2(tf.concat([x] + [m(x) for m in self.m], 3))


class TFSPPF(keras.layers.Layer):
    # Spatial pyramid pooling-Fast layer
    def __init__(self, c1, c2, k=5, w=None):
        """
        Initializes a fast spatial pyramid pooling layer with customizable in/out channels, kernel size, and weights.

        Args:
          c1 (int): Number of input channels.
          c2 (int): Number of output channels.
          k (int, optional): Kernel size for max pooling. Defaults to 5.
          w (object, optional): Pretrained weights for initializing the layer.

        Returns:
          None: This is an initializer method and does not return any value.

        Note:
          This class is part of the TensorFlow, Keras, and TFLite versions of YOLOv5. See `TFSPPF` initialization for details.

        Example:
          ```python
          layer = TFSPPF(c1=64, c2=128, k=5, w=pretrained_weights)
          ```
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_ * 4, c2, 1, 1, w=w.cv2)
        self.m = keras.layers.MaxPool2D(pool_size=k, strides=1, padding="SAME")

    def call(self, inputs):
        """
        Performs the forward pass on input data through the Spatial Pyramid Pooling-Fast layer, concatenating
        intermediate max-pooled results and applying final convolutions.

        Args:
          inputs (tf.Tensor): A 4D tensor with shape (batch, height, width, channels) representing input data.

        Returns:
          tf.Tensor: A 4D tensor containing the processed output, with shape determined by the SPPF layer configuration.

        Example usage:
        ```python
        # Assuming `inputs` is a properly shaped 4D tensor
        tfspf = TFSPPF(c1=128, c2=256, k=5)
        output = tfspf.call(inputs)
        ```
        """
        x = self.cv1(inputs)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(tf.concat([x, y1, y2, self.m(y2)], 3))


class TFDetect(keras.layers.Layer):
    # TF YOLOv5 Detect layer
    def __init__(self, nc=80, anchors=(), ch=(), imgsz=(640, 640), w=None):
        """
        Initialize the YOLOv5 detection layer for TensorFlow models.

        Args:
            nc (int): Number of classes.
            anchors (tuple): Anchor boxes for the detection layers.
            ch (tuple[int]): Number of input channels for each detection layer.
            imgsz (tuple[int, int]): Input image size, typically (height, width).
            w (object): Pretrained weights and model configuration.

        Returns:
            None

        Note: This detection layer is designed to integrate with the YOLOv5 TensorFlow model, facilitating object detection by
              computing various features such as stride, anchor grid, and grid cell configuration. This initialization sets all
              necessary attributes for forward passes during inference and training.
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
        Perform a forward pass through the YOLOv5 detection layer to predict object bounding boxes and classifications.

        Args:
            inputs (list[tf.Tensor]): List of input tensors from previous layer with shapes matching the expected
                                      input shape of the model.

        Returns:
            list[tf.Tensor]: Detection outputs containing bounding box information and class scores for each layer.
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
        Generates a 2D grid of (x, y) coordinates for YOLO detection layer.

        Args:
            nx (int): Number of grid cells along the x-axis. Default is 20.
            ny (int): Number of grid cells along the y-axis. Default is 20.

        Returns:
            tf.Tensor: A tensor representing a grid of shape [1, 1, ny*nx, 2] containing (x, y) coordinates.

        Examples:
            ```python
            grid = TFDetect._make_grid(nx=20, ny=20)
            ```
        """
        # return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
        xv, yv = tf.meshgrid(tf.range(nx), tf.range(ny))
        return tf.cast(tf.reshape(tf.stack([xv, yv], 2), [1, 1, ny * nx, 2]), dtype=tf.float32)


class TFSegment(TFDetect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), imgsz=(640, 640), w=None):
        """
        Initializes a YOLOv5 Segment head for segmentation models, supporting detection and mask prediction.

        Args:
            nc (int): Number of classes.
            anchors (tuple): Anchor configurations for bounding box predictions.
            nm (int): Number of mask outputs.
            npr (int): Number of protos.
            ch (tuple): Input channels.
            imgsz (tuple): Input image size in the form (height, width).
            w: Pretrained weights for the model layers.

        Returns:
            None

        Example:
            ```python
            model = TFSegment(nc=80, anchors=anchors, nm=32, npr=256, ch=(256, 512, 1024), imgsz=(640, 640), w=weights)
            ```
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
        Processes input through detection and segmentation layers, combining predictions into a unified output tensor.

        Args:
         x (tf.Tensor): Input tensor representing images for segmentation and detection. It's a list where each
                        element is a tensor corresponding to a specific level in the feature pyramid.

        Returns:
         tf.Tensor | tuple: Depending on the training mode, returns either raw feature map tensors during training
                            or concatenated detection outputs and proto masks during inference.

        Example:
        ```python
        from ultralytics.models.tf import TFSegment

        # Initialize a segmentation model
        model = TFSegment(nc=80, anchors=[...], ch=[...], imgsz=(640, 640), w=...)

        # Perform forward pass
        detections, proto_masks = model(images)
        ```
        """
        p = self.proto(x[0])
        # p = TFUpsample(None, scale_factor=4, mode='nearest')(self.proto(x[0]))  # (optional) full-size protos
        p = tf.transpose(p, [0, 3, 1, 2])  # from shape(1,160,160,32) to shape(1,32,160,160)
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p)


class TFProto(keras.layers.Layer):
    def __init__(self, c1, c_=256, c2=32, w=None):
        """
        Initializes a prototype head for segmentation with a set of convolutional and upsampling layers.

        Args:
            c1 (int): Number of input channels.
            c_ (int, optional): Number of intermediate channels. Defaults to 256.
            c2 (int, optional): Number of output channels. Defaults to 32.
            w (object, optional): Pretrained weights for the model layers.

        Returns:
            None

        Note:
            This layer forms part of the segmentation head in YOLOv5 models, helping to generate high-resolution feature maps
            used in segmentation masks.

        Example:
            ```python
            proto_layer = TFProto(c1=128, c_=256, c2=32)
            result = proto_layer(input_tensor)
            ```
        """
        super().__init__()
        self.cv1 = TFConv(c1, c_, k=3, w=w.cv1)
        self.upsample = TFUpsample(None, scale_factor=2, mode="nearest")
        self.cv2 = TFConv(c_, c_, k=3, w=w.cv2)
        self.cv3 = TFConv(c_, c2, w=w.cv3)

    def call(self, inputs):
        """
        Processes input tensor through convolutional layers and upscaling, producing transformed feature tensor.

        Args:
            inputs (tf.Tensor): Input tensor with shape [batch_size, height, width, channels] to be processed.

        Returns:
            tf.Tensor: Output tensor after applying convolution and upsample layers.
        """
        return self.cv3(self.cv2(self.upsample(self.cv1(inputs))))


class TFUpsample(keras.layers.Layer):
    # TF version of torch.nn.Upsample()
    def __init__(self, size, scale_factor, mode, w=None):
        """
        Initializes a TensorFlow upsampling layer with specified parameters.

        Args:
            size (int | tuple | None): Desired output size (height, width). Use `None` to specify using `scale_factor`.
            scale_factor (int | float): Multiplicative factor for input size. Must be a multiple of 2.
            mode (str): Interpolation mode, must be 'nearest'.
            w (Optional[torch.nn.Module]): Model weights, not used in this layer.

        Returns:
            None
        Notes:
            Ensures the `scale_factor` is a multiple of 2 for consistency in upsampling operations.

        Example:
            ```python
            upsample_layer = TFUpsample(size=None, scale_factor=2, mode='nearest')
            output = upsample_layer(input_tensor)
            ```
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
        Applies upsample operation to input tensors using specified interpolation mode.

        Args:
            inputs (tf.Tensor): Input tensor to be upsampled.

        Returns:
            tf.Tensor: Upsampled tensor.

        Examples:
            ```python
            import tensorflow as tf
            from some_module import TFUpsample

            upsample_layer = TFUpsample(size=None, scale_factor=4, mode='nearest')
            input_tensor = tf.random.normal([1, 64, 64, 3])  # Example input tensor
            output_tensor = upsample_layer(input_tensor)
            print(output_tensor.shape)  # Should print (1, 256, 256, 3) if scale_factor=4
            ```

        Notes:
            The scale factor should always be a multiple of 2 to avoid inconsistencies in upsampling.
            Ensure that all necessary arguments, including 'w', are provided during initialization.
        """
        return self.upsample(inputs)


class TFConcat(keras.layers.Layer):
    # TF version of torch.concat()
    def __init__(self, dimension=1, w=None):
        """
        Initializes the TFConcat layer, which mimics PyTorch's concatenation but for NCHW-to-NHWC format conversion.

        Args:
            dimension (int): Specifies the concatenation axis. Must be 1 for NCHW-to-NHWC conversion.
            w (Any): Placeholder for consistency with PyTorch API, not used in this implementation.

        Returns:
            None: This is an initializer method and does not return a value.

        Raises:
            AssertionError: If the dimension is not 1.

        Notes:
            This layer specifically handles the concatenation for converting NCHW format (channels first) to NHWC format
            (channels last), a common necessity in translating PyTorch models to TensorFlow.
        ```python
        # Example usage:
        concat_layer = TFConcat(dimension=1)
        ```
        """
        super().__init__()
        assert dimension == 1, "convert only NCHW to NHWC concat"
        self.d = 3

    def call(self, inputs):
        """
        Concatenates a list of tensors along the last dimension (axis 3).

        Args:
            inputs (list[tf.Tensor]): List of TensorFlow tensors to be concatenated.

        Returns:
            tf.Tensor: Concatenated tensor along the last dimension.
        """
        return tf.concat(inputs, self.d)


def parse_model(d, ch, model, imgsz):
    """
    Parses a model definition dictionary to create YOLOv5 model layers, including dynamic channel adjustments.

    Args:
        d (dict): Model definition dictionary containing architecture configuration.
        ch (list[int]): List of input channel dimensions.
        model (Model): Base model containing weights and structure.
        imgsz (tuple[int, int]): Image size (width, height) to be used.

    Returns:
        keras.Model: TensorFlow model with specified layers and configurations.

    Notes:
        This function dynamically constructs model layers based on a dictionary that defines the architecture of the YOLOv5 model.
        It adapts channel dimensions and other parameters to allow for flexible model configurations. The function evaluates string definitions of layers and integrates them into the model structure.

    Example:
        ```python
        model_dict = {
            "anchors": [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]],
            "nc": 80,
            "depth_multiple": 0.33,
            "width_multiple": 0.50,
            "backbone": [
                [-1, 1, 'Conv', [64, 3, 1]],
                [-1, 1, 'BottleneckCSP', [128, 3]],
                [-1, 1, 'SPP', [256]],
            ],
            "head": [
                [-1, 1, 'Conv', [512, 1, 1]],
                [-1, 1, 'Detect', [80, 3]],
            ],
        }
        channels = [3]
        model_instance = SomeModelClass()  # This should be an instance of the model containing weights and structure
        image_size = (640, 640)
        keras_model = parse_model(model_dict, channels, model_instance, image_size)
        ```

    References:
        Further information and updates on YOLOv5 can be found at:
        - https://github.com/ultralytics/yolov5
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
        __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, model=None, imgsz=(640, 640))

        Initializes a TensorFlow YOLOv5 model with a given configuration, number of input channels, class count, model
        instance, and input image size.

        Args:
            cfg (str | dict): Configuration file path or dictionary containing model configuration. Defaults to "yolov5s.yaml".
            ch (int): Number of input channels. Defaults to 3.
            nc (int, optional): Number of object classes. If provided, this will override the class count in the cfg.
            model (torch.nn.Module, optional): Predefined PyTorch model to be converted into TensorFlow model. If None, model
                                              will be created from cfg. Defaults to None.
            imgsz (tuple(int, int)): Input image size (width, height). Defaults to (640, 640).

        Returns:
            None

        Notes:
            - If the `cfg` argument is a string, it reads the YAML file specified by the path.
            - The model is defined based on the configuration and input arguments.
            - If `nc` is provided and differs from the class count in the `cfg`, the class count in the `cfg` will be
              overridden with `nc`.
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
        Runs inference on input data using the TensorFlow YOLOv5 model, providing options for applying TensorFlow Non-
        Maximum Suppression (NMS).

        Args:
            inputs (tf.Tensor): Input tensor for the model.
            tf_nms (bool): If True, applies TensorFlow NMS to filter the predicted bounding boxes. Default is False.
            agnostic_nms (bool): If True, the NMS process will ignore class labels during the suppression process. Default is False.
            topk_per_class (int): The maximum number of top-scoring boxes to keep per class before NMS for each image. Default is 100.
            topk_all (int): The maximum number of top-scoring boxes to keep overall before NMS for each image. Default is 100.
            iou_thres (float): Intersection-over-union (IoU) threshold for NMS. Default is 0.45.
            conf_thres (float): Confidence threshold for filtering weak predictions before NMS. Default is 0.25.

        Returns:
            tuple: Containing the following elements:
                - nms (tf.Tensor): A tensor of boxes after NMS if `tf_nms` is True.
                - y (list of tf.Tensor): A list of layer outputs. Only layers specified in `savelist` will have their outputs saved.

        Example:
            ```python
            # Example of running prediction with the model
            model = TFModel(cfg='yolov5s.yaml', ch=3)
            img = tf.random.uniform((1, 640, 640, 3))
            predictions = model.predict(img, tf_nms=True)

            # To get outputs directly from model without NMS
            outputs = model.predict(img, tf_nms=False)
            ```

        Note:
            This function runs through all layers defined in the model and optionally applies TensorFlow NMS for further filtering of bounding boxes.
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
            xywh (tf.Tensor): A tensor of shape (..., 4) containing bounding boxes in [x, y, w, h] format, where x and y
                              represent the center of the box, and w and h are the width and height respectively.

        Returns:
            tf.Tensor: A tensor of the same shape as `xywh`, containing bounding boxes in [x1, y1, x2, y2] format, where
                       (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner of the bounding box.

        Example:
            ```python
            boxes_xywh = tf.constant([[50, 50, 20, 30]], dtype=tf.float32)
            boxes_xyxy = TFModel._xywh2xyxy(boxes_xywh)
            ```

        Note:
            This function is typically used to transform bounding box coordinates for easier calculation of areas and
            intersection-over-union (IoU).
        """
        x, y, w, h = tf.split(xywh, num_or_size_splits=4, axis=-1)
        return tf.concat([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)


class AgnosticNMS(keras.layers.Layer):
    # TF Agnostic NMS
    def call(self, input, topk_all, iou_thres, conf_thres):
        """
        AgnosticNMS.call(input, topk_all, iou_thres, conf_thres)

        Executes the agnostic Non-Maximum Suppression (NMS) on given input tensors. This method is designed to suppress overlapping bounding boxes, irrespective of their class.

        Args:
        input (tuple): A tuple containing boxes, classes, and scores tensors.
          - boxes (tf.Tensor): A tensor of shape [batch, num_boxes, 4] containing coordinates of the bounding boxes.
          - classes (tf.Tensor): A tensor of shape [batch, num_boxes] containing class indices for each box.
          - scores (tf.Tensor): A tensor of shape [batch, num_boxes, num_classes] containing class scores for each box.
        topk_all (int): The maximum number of boxes to keep after NMS.
        iou_thres (float): Intersection Over Union (IoU) threshold for NMS.
        conf_thres (float): Confidence score threshold for filtering boxes.

        Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: A tuple containing:
          - nmsed_boxes (tf.Tensor): The filtered bounding boxes after NMS.
          - nmsed_scores (tf.Tensor): The scores of the filtered boxes.
          - nmsed_classes (tf.Tensor): The class indices of the filtered boxes.
          - nmsed_valid_detections (tf.Tensor): The number of valid detections in each batch.

        Notes:
        This method works agnostically with respect to the class of the boxes, meaning that it does not differentiate between different classes during suppression.
        The input tensors should be properly formatted to ensure correct operation.
        The method leverages `tf.map_fn` for efficient processing of input batches.

        Example:
        ```python
        boxes = tf.constant([[[0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.4, 0.4]]]) # shape [1, 2, 4]
        classes = tf.constant([[0, 1]]) # shape [1, 2]
        scores = tf.constant([[[0.9, 0.1], [0.75, 0.25]]]) # scores for 2 classes
        nms_output = AgnosticNMS().call((boxes, classes, scores), topk_all=1, iou_thres=0.5, conf_thres=0.25)
        ```
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
        Performs agnostic non-maximum suppression (NMS) on detected objects, filtering based on IoU and confidence
        thresholds.

        Args:
          x (tf.Tensor): A tensor containing detected boxes, classes, and scores with shape `[n, 3]`.
          topk_all (int): Maximum number of detections to keep after applying NMS.
          iou_thres (float): Intersection over Union (IoU) threshold for filtering overlapping bounding boxes.
          conf_thres (float): Confidence threshold for filtering low-confidence detections.

        Returns:
          Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: A tuple containing:
              - padded_boxes (tf.Tensor): Padded tensor of selected bounding boxes with shape `[topk_all, 4]`.
              - padded_scores (tf.Tensor): Padded tensor of selected confidence scores with shape `[topk_all]`.
              - padded_classes (tf.Tensor): Padded tensor of selected class indices with shape `[topk_all]`.
              - valid_detections (tf.Tensor): Scalar tensor indicating the number of valid detections.

        Note:
          This function performs per-image NMS processing. It uses TensorFlow operations to manage padding and ensure
          the output conforms with the pre-defined maximum number of detections.

        Example:
        ```python
        boxes, classes, scores = tf.random.normal([100, 4]), tf.random.uniform([100, 80]), tf.random.uniform([100, 1])
        nms_results = AgnosticNMS._nms((boxes, classes, scores), topk_all=50, iou_thres=0.45, conf_thres=0.25)
        ```
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
    Converts PyTorch activation functions to their TensorFlow equivalents.

    Args:
        act (nn.Module): The activation function from PyTorch, such as nn.LeakyReLU, nn.Hardswish, or nn.SiLU/Swish.

    Returns:
        Callable: A TensorFlow-compatible activation function.

    Examples:
        ```python
        import torch.nn as nn

        # Convert PyTorch nn.LeakyReLU to TensorFlow equivalent
        tf_activation = activations(nn.LeakyReLU())
        ```

    Notes:
        - Supported PyTorch activation functions include nn.LeakyReLU, nn.Hardswish, and nn.SiLU (also known as Swish).
        - Acts as a bridge for converting activation functions in a model trained with PyTorch to TensorFlow for further processing.
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
    Generates a representative dataset for model calibration with TensorFlow Lite.

    Args:
        dataset (object): The input dataset, which typically iterates over paths, images, original images, video captures,
            and strings.
        ncalib (int): The number of calibration iterations to perform. Defaults to 100.

    Yields:
        list[np.ndarray]: Transformed image arrays suitable for TensorFlow Lite model calibration.

    Example:
        ```python
        representative_data = representative_dataset_gen(dataset)
        for data in representative_data:
            interpreter.set_tensor(input_details[0]['index'], data)
            interpreter.invoke()
        ```

    The generated representative data is essential for quantizing models, ensuring they are optimized for efficiency
    without significant loss in accuracy.
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
    Run a YOLOv5 model export from PyTorch to TensorFlow and Keras formats, with optional inference and validation.

    Args:
        weights (str | pathlib.Path): Path to the weights file.
        imgsz (tuple[int, int]): Inference size in height and width.
        batch_size (int): Number of images per batch.
        dynamic (bool): Whether to use a dynamic batch size.

    Returns:
        None

    Example:
        ```python
        run(weights="path/to/yolov5s.pt", imgsz=(640, 640), batch_size=1, dynamic=False)
        ```
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
    Parses and returns command-line options for model inference, including weights path, image size, batch size, and
    dynamic batching.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed command-line arguments encapsulated in an argparse.Namespace object.

    Example:
        ```python
        if __name__ == "__main__":
            opt = parse_opt()
            print(opt.weights, opt.imgsz, opt.batch_size, opt.dynamic)
        ```

    Notes:
        Default weights file path is `ROOT / "yolov5s.pt"`. When providing an image size, if only one value is specified, it will be duplicated to form a square. The dynamic argument is a flag and does not take a value. It adjusts the model to handle dynamic batch sizes.

    Related:
        See [print_args](https://github.com/ultralytics/yolov5/blob/master/utils/general.py#L30) for argument printing.
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
    Main function to run YOLOv5 model with parsed command-line options.

    Args:
     opt (Namespace): Parsed command-line arguments.

    Returns:
     None

    Examples:
     ```python
     if __name__ == "__main__":
         opt = parse_opt()
         main(opt)
     ```

    Notes:
     This function serves as the entry point for running the YOLOv5 model. It uses the
     options specified via command-line arguments to configure and execute a model run.
     For more usage information, visit https://github.com/ultralytics/yolov5.
    ```
    """
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
