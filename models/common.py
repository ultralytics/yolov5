# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Common modules."""

import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

# Import 'ultralytics' package or install if missing
try:
    import ultralytics

    assert hasattr(ultralytics, "__version__")  # verify package is not directory
except (ImportError, AssertionError):
    import os

    os.system("pip install -U ultralytics")
    import ultralytics

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from utils import TryExcept
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (
    LOGGER,
    ROOT,
    Profile,
    check_requirements,
    check_suffix,
    check_version,
    colorstr,
    increment_path,
    is_jupyter,
    make_divisible,
    non_max_suppression,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
    yaml_load,
)
from utils.torch_utils import copy_attr, smart_inference_mode


def autopad(k, p=None, d=1):
    """
    Perform kernel padding to maintain 'same' output shape, adjusting for optional dilation; returns padding size.

    Args:
        k (int | list[int]): The size of the kernel, which could be an integer or a list of integers.
        p (int | list[int], optional): Padding value (integer or list). If `None`, automatically calculates padding.
        d (int, optional): Dilation rate (default is 1).

    Returns:
        (int | list[int]): Calculated padding size, matching the type of the kernel `k`.

    Example:
        ```python
        # Example with integer kernel size
        padding = autopad(3)

        # Example with list kernel size and dilation
        padding = autopad([3, 5], d=2)
        ```

    Note:
        The function computes the padding size required to maintain the same output shape for a given kernel size and
        optional dilation. If padding `p` is not provided, it defaults to 'same' padding calculation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize a standard convolutional layer with batch normalization and optional activation.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int | list[int], optional): Kernel size. Defaults to 1.
            s (int, optional): Stride. Defaults to 1.
            p (int | list[int] | None, optional): Padding. If `None`, it is automatically calculated to achieve 'same' output
                shape. Defaults to `None`.
            g (int, optional): Number of groups in Grouped Convolutions. Defaults to 1.
            d (int, optional): Dilation. Defaults to 1.
            act (bool, optional): Whether to include an activation function. Defaults to `True`.

        Returns:
            (torch.nn.Module): A convolutional layer with batch normalization and optional activation.

        Example:
            ```python
            conv_layer = Conv(32, 64, k=3, s=1, p=1)
            output = conv_layer(input_tensor)
            ```
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Processes the input tensor `x` using convolution, batch normalization, and activation function.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W), where N is the batch size, C is the number of channels,
                H is the height, and W is the width.

        Returns:
            (torch.Tensor): Output tensor after applying convolution, batch normalization, and the activation function.

        Example:
            ```python
            import torch
            from ultralytics import Conv

            # Initialize Conv layer
            conv_layer = Conv(c1=3, c2=16, k=3, s=1, p=1, act=True)

            # Create a sample input tensor with shape (8, 3, 224, 224)
            x = torch.rand(8, 3, 224, 224)

            # Forward pass
            output = conv_layer(x)
            print(output.shape)  # Expected output tensor shape (8, 16, 224, 224)
            ```
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Fuses convolution and activation layers for an optimized forward pass.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            (torch.Tensor): Output tensor after fusing convolution and activation layers, with shape (N, C_out, H_out, W_out).

        Example:
            ```python
            x = torch.randn(1, 3, 224, 224)  # example input tensor
            conv_layer = Conv(3, 64, k=3, s=1)
            y = conv_layer.forward_fuse(x)
            ```
        """
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """
        Initializes a depth-wise convolutional layer with optional activation.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size. Default is 1.
            s (int): Stride size. Default is 1.
            d (int): Dilation factor. Default is 1.
            act (bool | nn.Module): If True, applies the default activation function. If an instance of nn.Module,
                applies the specified activation. Default is True.

        Returns:
            None

        Example:
            ```python
            # Initialize a depth-wise convolution layer with 3 input channels, 64 output channels, and default parameters
            dwconv_layer = DWConv(3, 64)

            # Initialize a depth-wise convolution layer with custom parameters
            dwconv_layer_custom = DWConv(3, 64, k=3, s=2, d=1, act=nn.ReLU())
            ```
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """
        Initialize a depth-wise transpose convolutional layer for YOLOv5.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Size of the convolution kernel. Defaults to 1.
            s (int): Stride of the convolution. Defaults to 1.
            p1 (int): Input padding. Defaults to 0.
            p2 (int): Output padding. Defaults to 0.

        Returns:
            (torch.Tensor): Output tensor after applying depth-wise transpose convolution.

        Example:
            ```python
            # Initialize the layer
            layer = DWConvTranspose2d(c1=128, c2=256, k=3, s=2, p1=1, p2=1)
            # Create a random input tensor with shape (batch_size, c1, height, width)
            x = torch.randn(1, 128, 64, 64)
            # Apply the layer
            output = layer(x)
            ```
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        """
        Initialize a transformer layer, sans LayerNorm for performance, with multihead attention and linear layers.

        Args:
            c (int): The number of channels, which determines the dimensionality of the input and output feature spaces.
            num_heads (int): The number of attention heads to be used in the multi-head attention mechanism, allowing for
                multiple representation subspaces.

        Returns:
            None

        Example:
            ```python
            layer = TransformerLayer(256, 4)
            input_tensor = torch.rand(10, 32, 256)  # (sequence_length, batch_size, channels)
            output = layer(input_tensor)
            ```

        Note:
            This implementation leverages mechanisms from https://arxiv.org/abs/2010.11929 but removes LayerNorm layers for better
            performance.
        """
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """
        Performs forward pass using attention mechanism and linear projections with residual connections.

        Args:
            x (torch.Tensor): Input tensor with shape (L, N, E), where L is the sequence length, N is the batch size,
                and E is the embedding dimension.

        Returns:
            (torch.Tensor): Output tensor with shape (L, N, E), transformed by the multi-head attention and linear layers with
                residual connections.

        Example:
            ```python
            layer = TransformerLayer(256, 4)
            input_tensor = torch.rand(10, 32, 256)  # (sequence_length, batch_size, channels)
            output = layer(input_tensor)
            ```

        Note:
            This implementation does not include LayerNorm layers for performance optimization. For more details, refer to
            [original paper](https://arxiv.org/abs/2010.11929).
        """
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        """
        Initialize a Transformer block for vision tasks, adapting dimensions if necessary and stacking specified layers.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels. If different from `c1`, a convolution layer will be used to adapt dimensions.
            num_heads (int): Number of attention heads in each Transformer layer.
            num_layers (int): Number of Transformer layers to stack.

        Returns:
            None

        Example:
            ```python
            transformer_block = TransformerBlock(c1=64, c2=128, num_heads=8, num_layers=6)
            inputs = torch.rand(32, 64, 64)  # Example input tensor with shape (batch_size, channels, sequence_length)
            outputs = transformer_block(inputs)
            ```

        Note:
            This block is based on the Vision Transformer architecture described in https://arxiv.org/abs/2010.11929.
        """
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """
        Processes input tensor through convolution (if necessary), then applies Transformer layers and position
        embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C1, W, H).

        Returns:
            (torch.Tensor): Transformed tensor with shape (B, C2, W, H), where C2 is the output channel size.

        Example:
            ```python
            transformer_block = TransformerBlock(c1=32, c2=64, num_heads=8, num_layers=6)
            input_tensor = torch.rand(8, 32, 32, 32)  # Batch of 8 images with 32 channels
            output_tensor = transformer_block(input_tensor)
            print(output_tensor.shape)  # Should print: torch.Size([8, 64, 32, 32])
            ```

        Note:
            This function assumes the input tensor has a suitable shape for the included convolution (if applicable) and
            transformer layers.
        """
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """
        Initialize a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            shortcut (bool): If True, add a residual connection.
            g (int): Number of convolution groups.
            e (float): Channel expansion ratio.

        Returns:
            (torch.Tensor): The transformed tensor after applying the bottleneck layer operations.

        Example:
            ```python
            from ultralytics.models.common import Bottleneck

            bottleneck = Bottleneck(c1=64, c2=128, shortcut=True, g=1, e=0.5)
            input_tensor = torch.rand(1, 64, 32, 32)  # Batch size 1, 64 channels, 32x32 spatial resolution
            output_tensor = bottleneck(input_tensor)
            ```
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Processes input through two convolutional layers and optionally applies a residual connection.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C1, H, W) to process.

        Returns:
            (torch.Tensor): Processed tensor of shape (N, C2, H, W), where N is batch size, C1 is input channels, and C2 is output
            channels.

        Example:
            ```python
            bottleneck_layer = Bottleneck(64, 128)
            input_tensor = torch.rand(1, 64, 32, 32)
            output_tensor = bottleneck_layer(input_tensor)
            print(output_tensor.shape)  # should output torch.Size([1, 128, 32, 32])
            ```

        Note:
            The residual connection is applied if `shortcut=True` and input and output channels match.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initializes a CSP (Cross Stage Partial) bottleneck module with multiple bottlenecks.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of repeated bottleneck layers. Defaults to 1.
            shortcut (bool): If True, includes shortcut connections within bottlenecks. Defaults to True.
            g (int): Number of groups for convolutions. Defaults to 1.
            e (float): Expansion factor for bottlenecks. Defaults to 0.5.

        Returns:
            None

        Example:
            ```python
            # Create a BottleneckCSP module with 64 input channels and 128 output channels, repeated 3 times
            bottleneck_csp = BottleneckCSP(64, 128, n=3)
            ```

        Note:
            The CSP bottleneck implementation is based on the paper available at https://github.com/WongKinYiu/CrossStagePartialNetworks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """
        Perform forward pass by applying sequential bottleneck layers, activations, and concatenations on input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) where N is batch size, C is number of channels, H and W
                are height and width of the input feature map, respectively.

        Returns:
            (torch.Tensor): Output tensor after processing through CSP bottleneck layers, with enhanced features.

        Example:
            ```python
            import torch
            from ultralytics.modules import BottleneckCSP

            # Create BottleneckCSP module with input channels 64, output channels 128, and 2 Bottleneck layers
            csp_bottleneck = BottleneckCSP(64, 128, n=2)
            input_tensor = torch.randn(1, 64, 256, 256)  # Example input tensor
            output_tensor = csp_bottleneck(input_tensor)
            print(output_tensor.shape)  # Should print torch.Size([1, 128, 256, 256])
            ```

        Note:
            This module is designed based on the Cross-Stage Partial Networks architecture:
            https://github.com/WongKinYiu/CrossStagePartialNetworks.
        """
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        """
        Initializes CrossConv with downsampling, expanding, and optionally shortcutting; `c1` input, `c2` output
        channels.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for convolutions. Default is 3.
            s (int): Stride for convolutions. Default is 1.
            g (int): Number of convolution groups. Default is 1.
            e (float): Expansion ratio for hidden channels. Default is 1.0.
            shortcut (bool): Whether to use a shortcut connection. Default is False.

        Note:
            The first convolution applies a kernel shape of `(1, k)` followed by `(k, 1)` in the second convolution
            to achieve cross convolutional down-sampling.

        Example:
            ```python
            cross_conv = CrossConv(64, 128, k=3, s=2, g=1, e=0.5, shortcut=True)
            output = cross_conv(input_tensor)
            ```
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Perform feature downsampling and expansion, optionally adding a shortcut if input-output channels match.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C1, H, W).

        Returns:
            (torch.Tensor): Processed tensor with shape (N, C2, H', W').

        Example:
            ```python
            import torch
            from ultralytics.modules import CrossConv

            cross_conv = CrossConv(64, 128, k=3, s=2, shortcut=True)
            input_tensor = torch.rand(1, 64, 32, 32)
            output_tensor = cross_conv(input_tensor)
            print(output_tensor.shape)  # Should print: torch.Size([1, 128, 16, 16])
            ```

        Note:
            The function combines two convolutions with kernel shapes `(1, k)` and `(k, 1)`, respectively, and can include
            shortcut connections if `shortcut` is set to True.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize the C3 module with options for channel count, bottleneck repetition, shortcut, group convolutions,
        and expansion.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of Bottleneck units to repeat.
            shortcut (bool): Use of shortcut connection; defaults to True.
            g (int): Number of groups for convolutions; typically for group convolutions.
            e (float): Expansion ratio for bottleneck channels; defaults to 0.5.

        Returns:
            (None): This constructor does not return a value.

        Example:
            ```python
            import torch
            from ultralytics import C3

            c3_module = C3(64, 128, n=3, shortcut=False)
            input_tensor = torch.rand(1, 64, 256, 256)
            output_tensor = c3_module(input_tensor)
            ```
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """
        Performs forward propagation utilizing concatenated features from convolutions and a Bottleneck sequence.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W), where B is batch size, C is number of channels,
                              H and W are height and width of the input feature map, respectively.

        Returns:
            (torch.Tensor): Output tensor after processing through the C3 module, with enhanced features and shape
                            (B, C2, H, W).

        Example:
            ```python
            import torch
            from ultralytics.models.common import C3

            # Initialize a C3 module
            c3_layer = C3(64, 128, n=3)
            x = torch.randn(1, 64, 32, 32)
            output = c3_layer(x)
            print(output.shape)  # Should print torch.Size([1, 128, 32, 32])
            ```
        """
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize the C3x module with cross-convolutions, extending C3 for additional feature extraction capabilities.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of bottleneck layers to stack. Defaults to 1.
            shortcut (bool): Whether to use shortcut connections. Defaults to True.
            g (int): Number of groups to use in grouped convolutions. Defaults to 1.
            e (float): Channel expansion ratio. Defaults to 0.5.

        Returns:
            None

        Example:
            ```python
            import torch
            from ultralytics.modules.common import C3x

            # Initialize the C3x module
            c3x_module = C3x(c1=64, c2=128, n=3, shortcut=True, g=1, e=0.5)
            # Example input with batch size of 1, 64 channels, 128x128 spatial dimensions
            input_tensor = torch.rand(1, 64, 128, 128)
            # Forward pass
            output_tensor = c3x_module(input_tensor)
            print(output_tensor.shape)  # Expected output shape: (1, 128, 128, 128)
            ```

        Notes:
            This module is an extension of the standard C3 module within the Ultralytics YOLO architecture, combining cross-layer
            convolutions with the CSP bottleneck structure to enhance feature extraction.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize a C3 module with Transformer blocks for enhanced feature extraction, utilizing convolution,
        bottlenecks, and transformer blocks.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of Bottleneck or Transformer layers to include. Defaults to 1.
            shortcut (bool): Whether to use shortcuts between layers. Defaults to True.
            g (int): Number of groups for group convolution. Defaults to 1.
            e (float): Expansion ratio for the hidden channels. Defaults to 0.5.

        Returns:
            None

        Example:
            ```python
            from ultralytics import C3TR

            model = C3TR(c1=64, c2=128, n=3, shortcut=True, g=1, e=0.5)
            output = model(input_tensor)
            ```

        Note:
            This module integrates Transformer blocks with Convolutional layers to facilitate advanced feature extraction
            capabilities, making it suitable for complex vision tasks.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        """
        Initializes a C3 module with an SPP (Spatial Pyramid Pooling) layer for advanced spatial feature extraction.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (tuple[int]): Kernel sizes for SPP layers, typically (5, 9, 13).
            n (int): Number of Bottleneck modules to stack. Defaults to 1.
            shortcut (bool): If True, enables shortcut connections within Bottleneck modules. Defaults to True.
            g (int): Number of groups for group convolution. Defaults to 1.
            e (float): Expansion ratio for hidden channels. Defaults to 0.5.

        Returns:
            (None): This constructor does not return a value.

        Example:
            ```python
            from ultralytics import C3SPP

            # Initialize a C3SPP module
            c3spp_module = C3SPP(c1=64, c2=128, k=(5, 9, 13), n=2, shortcut=True, g=1, e=0.5)
            input_tensor = torch.rand(1, 64, 256, 256)
            output_tensor = c3spp_module(input_tensor)
            print(output_tensor.shape)  # Should output torch.Size([1, 128, 256, 256])
            ```

        Note:
            The SPP layer enhances the receptive field by applying multiple max-pooling operations with varying kernel sizes.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize the C3 module using GhostBottleneck layers for efficient feature extraction.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of bottleneck layers to stack. Default is 1.
            shortcut (bool): Whether to use shortcut connections. Default is True.
            g (int): Number of groups for group convolution. Default is 1.
            e (float): Expansion ratio for hidden channels. Default is 0.5.

        Example:
            ```python
            import torch
            from ultralytics import C3Ghost

            c3ghost_module = C3Ghost(64, 128, n=3, shortcut=True, g=1, e=0.5)
            input_tensor = torch.rand(1, 64, 256, 256)
            output_tensor = c3ghost_module(input_tensor)
            ```
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        """
        Initialize an SPP (Spatial Pyramid Pooling) layer for feature map processing in convolutional neural networks.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (tuple[int]): Tuple of kernel sizes for the spatial pyramid pooling layers.

        Returns:
            None

        Example:
            ```python
            spp_layer = SPP(c1=256, c2=512, k=(5, 9, 13))
            input_tensor = torch.randn(1, 256, 32, 32)  # Example input tensor
            output_tensor = spp_layer(input_tensor)
            print(output_tensor.shape)  # Expected shape: torch.Size([1, 512, 32, 32])
            ```

        Note:
            Spatial Pyramid Pooling (SPP) layers are used to generate fixed-length representations regardless of input size by
            applying multiple pooling operations with different kernel sizes.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """
        Forward pass through the Spatial Pyramid Pooling (SPP) layer, performing convolution and max pooling.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) where N is the batch size, C is the number of channels,
                H is the height, and W is the width of the input feature map.

        Returns:
            (torch.Tensor): Output tensor after applying convolution and spatial pyramid pooling, with shape (N, C_out, H, W)
                where C_out is the number of output channels.

        Example:
            ```python
            import torch
            from ultralytics.models.common import SPP

            # Initialize SPP module with 128 input channels and 256 output channels
            spp = SPP(128, 256)

            # Create a random input tensor with shape (16, 128, 32, 32)
            input_tensor = torch.randn(16, 128, 32, 32)

            # Perform forward pass
            output_tensor = spp.forward(input_tensor)

            print(output_tensor.shape)  # Should print: torch.Size([16, 256, 32, 32])
            ```
        """
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):
        """
        Initializes YOLOv5 Spatial Pyramid Pooling - Fast (SPPF) layer for efficient feature extraction.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for the max pooling layer, default is 5.

        Notes:
            This module applies a series of three max-pooling operations followed by two convolutional operations. It is
            optimized for speed compared to the standard SPP layer with kernel sizes (5, 9, 13), achieving similar results
            while reducing computation time.

        Example:
            ```python
            from ultralytics.models.common import SPPF

            # Initialize SPPF layer
            sppf_layer = SPPF(c1=512, c2=1024)
            # Create a dummy input tensor with shape (batch_size, channels, height, width)
            x = torch.randn(1, 512, 20, 20)
            # Perform forward pass through the SPPF layer
            output = sppf_layer(x)
            print(output.shape)  # Should output: torch.Size([1, 1024, 20, 20])
            ```
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """
        Forward pass through the Spatial Pyramid Pooling - Fast (SPPF) layer for efficient feature extraction.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W), where N is the batch size, C is the number of channels,
                and H, W are height and width, respectively.

        Returns:
            (torch.Tensor): Output tensor after applying convolution and max pooling layers, concatenated along the channel dimension,
                with shape (N, C_out, H, W) where C_out is the number of output channels.

        Example:
            ```python
            from ultralytics.models.common import SPPF
            import torch

            sppf_layer = SPPF(c1=512, c2=1024)
            input_tensor = torch.randn(1, 512, 20, 20)  # Example input with batch size 1, and 512 channels
            output_tensor = sppf_layer(input_tensor)
            print(output_tensor.shape)  # Should output torch.Size([1, 1024, 20, 20])
            ```

        Note:
            This layer is optimized for speed compared to traditional SPP layers, combining convolutions and max pooling within
            a specified kernel size to improve processing efficiency.
        """
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        Initialize the Focus module to concentrate width-height information into channel space.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int, optional): Kernel size for the convolution operation. Default is 1.
            s (int, optional): Stride for the convolution operation. Default is 1.
            p (int, optional): Padding for the convolution operation. If None, uses autopad. Default is None.
            g (int, optional): Number of groups for the convolution operation. Default is 1.
            act (bool | nn.Module, optional): Activation function applied after convolution. If True, uses default activation.
                Default is True.

        Returns:
            (None): This constructor does not return any value.

        Example:
            ```python
            focus_layer = Focus(c1=3, c2=64)
            ```
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Processes input tensor through the Focus module for enhanced feature extraction.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W), where N is the batch size, C is the number of channels,
                and H and W are the height and width respectively.

        Returns:
            (torch.Tensor): Output tensor after transforming the input tensor, with enriched channel dimension.

        Example:
            ```python
            import torch
            from ultralytics import Focus

            focus_layer = Focus(c1=3, c2=64)
            input_tensor = torch.randn(1, 3, 640, 640)  # Batch of 1, 3 input channels, 640x640 image size
            output_tensor = focus_layer(input_tensor)
            print(output_tensor.shape)  # Expected shape (1, 64, 320, 320)
            ```
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """
        Initialize a Ghost Convolution layer for efficient feature extraction in neural networks.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int, optional): Kernel size. Defaults to 1.
            s (int, optional): Stride size. Defaults to 1.
            g (int, optional): Number of groups for grouped convolution. Defaults to 1.
            act (bool | nn.Module, optional): Activation function or a boolean indicating whether to use the default activation function. Defaults to True.

        Returns:
            None

        Example:
            ```python
            from ultralytics.models.common import GhostConv
            import torch

            # Initialize the GhostConv layer
            ghost_conv = GhostConv(64, 128, 3, 1)

            # Create a random input tensor of shape (batch_size, c1, height, width)
            input_tensor = torch.randn(1, 64, 224, 224)

            # Apply the GhostConv layer
            output_tensor = ghost_conv(input_tensor)
            print(output_tensor.shape)  # Expected output shape will be (1, 128, 224, 224)
            ```

        Note:
            This implementation is based on the GhostNet architecture from https://github.com/huawei-noah/ghostnet and
            optimizes computational efficiency by halving the number of output channels through concatenation of two
            convolution layers.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """
        Performs forward pass through Ghost Convolution module for efficient feature extraction.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W), where N is batch size, C is number of channels,
                              H and W are height and width respectively.

        Returns:
            (torch.Tensor): Output tensor after processing through Ghost Convolution layers, with enhanced efficiency
                            and reduced computation, maintaining shape (N, C_out, H, W).

        Example:
            ```python
            import torch
            from ultralytics.yolo_v5 import GhostConv

            ghost_conv = GhostConv(64, 128, 3, 1)
            input_tensor = torch.randn(1, 64, 224, 224)
            output_tensor = ghost_conv(input_tensor)
            ```

        Note:
            The Ghost Convolution module combines pointwise and depthwise convolutions followed by concatenation, inspired
            by https://github.com/huawei-noah/ghostnet to achieve efficient computation.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):
        """
        Initialize a GhostBottleneck layer with depthwise and pointwise convolutions for efficient feature extraction.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int, optional): Kernel size for depthwise convolution. Default is 3.
            s (int, optional): Stride for depthwise convolution. Default is 1.

        Returns:
            None

        Example:
            ```python
            from ultralytics.models.common import GhostBottleneck

            ghost_bottleneck = GhostBottleneck(c1=64, c2=128)
            input_tensor = torch.randn(1, 64, 32, 32)  # Batch size of 1, 64 input channels, 32x32 image size
            output_tensor = ghost_bottleneck(input_tensor)
            print(output_tensor.shape)  # Should print: torch.Size([1, 128, 32, 32])
            ```

        Note:
            This layer utilizes Ghost Convolution layers for pointwise operations and optionally depthwise convolutions for
            efficient feature extraction, inspired by GhostNet (https://github.com/huawei-noah/ghostnet).
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),
        )  # pw-linear
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """
        Processes input tensor through sequential GhostNet convolution layers and optional shortcut.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W) where N is the batch size, C is the number of channels,
                H and W are the height and width of the input feature map.

        Returns:
            (torch.Tensor): Output tensor after processing, with the same or altered shape depending on the stride of the
                convolutions.

        Example:
            ```python
            from ultralytics import GhostBottleneck
            import torch

            ghost_bottleneck = GhostBottleneck(c1=64, c2=128, k=3, s=1)
            input_tensor = torch.randn(1, 64, 56, 56)  # Shape (batch_size, channels, height, width)
            output_tensor = ghost_bottleneck(input_tensor)
            print(output_tensor.shape)  # Output shape should match the expected transformation
            ```

        Note:
            The function harnesses the GhostNet architecture, enabling efficient feature extraction with reduced computational
            complexity, as detailed in https://github.com/huawei-noah/ghostnet.
        """
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        """
        Initialize a layer to contract spatial dimensions (width-height) into channels.

        Args:
            gain (int): The factor by which to reduce the spatial dimensions, while increasing the number of channels.
                For instance, a gain of 2 will contract input shape from (B, C, H, W) to (B, C * gain^2, H / gain, W / gain).

        Returns:
            None

        Example:
            ```python
            contract_layer = Contract(gain=2)
            output = contract_layer(input_tensor)
            ```
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """
        Forward pass for the Contract layer.

        This method contracts the spatial dimensions of the input tensor (height and width) into the channel dimension by
        a factor specified by the `gain` parameter.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W), where B is the batch size, C is the number of channels,
                H is the height, and W is the width.

        Returns:
            (torch.Tensor): Output tensor with shape (B, C * gain^2, H // gain, W // gain), where `gain` is the contraction
                factor specified in the initialization.

        Example:
            ```python
            import torch
            from ultralytics.models.common import Contract

            x = torch.randn(1, 64, 80, 80)  # Example input tensor
            contract_layer = Contract(gain=2)
            result = contract_layer.forward(x)
            print(result.shape)  # should output (1, 256, 40, 40)
            ```

        Note:
            The Contract layer is useful in neural network architectures where reducing spatial resolution while increasing
            the channel count is beneficial for certain operations.
        """
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        """
        Initialize the Expand module to increase spatial dimensions by redistributing channels, with an optional gain
        factor.

        Args:
            gain (int): Factor by which to increase the spatial dimensions. Default is 2.

        Example:
            ```python
            x = torch.randn(1, 64, 80, 80)
            expand = Expand(gain=2)
            y = expand(x)  # shape: (1, 16, 160, 160)
            ```

        Note:
            The Expand module increases the width and height of the input tensor by reducing the channels accordingly.
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """
        Processes input tensor `x` to expand spatial dimensions by redistributing channels, effectively reversing the
        operation done by the `Contract` module.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W) where C should be divisible by `gain^2`.

        Returns:
            (torch.Tensor): Output tensor with expanded spatial dimensions and redistributed channels, with shape (B, C/gain^2,
            H*gain, W*gain).

        Example:
            ```python
            x = torch.randn(1, 64, 80, 80)  # Input tensor
            expand = Expand(gain=2)
            y = expand(x)
            print(y.shape)  # Output shape: (1, 16, 160, 160)
            ```

        Note:
            The `Expand` module is typically utilized in advanced feature extraction processes where spatial dimensions need to
            be increased efficiently without losing meaningful features extracted during contraction.
        """
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s**2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s**2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        """
        Concatenate a list of tensors along a specified dimension.

        Args:
            dimension (int): The dimension along which to concatenate the input tensors. Default is 1.

        Returns:
            None: This constructor does not return a value.

        Example:
            ```python
            # Create Concat module to concatenate along second dimension
            concat = Concat(dimension=1)

            # Define example tensors to concatenate
            tensor1 = torch.rand(1, 64, 128, 128)
            tensor2 = torch.rand(1, 64, 128, 128)
            tensor3 = torch.rand(1, 64, 128, 128)

            # Concatenate tensors
            output = concat([tensor1, tensor2, tensor3])
            print(output.shape)  # Should print torch.Size([1, 192, 128, 128])
            ```

        Notes:
            - Concatenation is performed along the specified dimension, allowing for flexible combination of input tensors with
              compatible shapes.
            - Ensure that the input tensors have matching sizes except for the specified dimension to avoid shape mismatch
              errors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """
        Concatenate a list of tensors along a specified dimension.

        Args:
            x (list[torch.Tensor]): List of tensors to be concatenated along the specified dimension.

        Returns:
            (torch.Tensor): Tensor obtained by concatenating the input tensors along the specified dimension.

        Example:
            ```python
            import torch
            from ultralytics.modules import Concat

            concat = Concat(dimension=1)
            tensors = [torch.randn(1, 2, 3), torch.randn(1, 2, 3)]
            result = concat(tensors)
            print(result.shape)  # Output: torch.Size([1, 4, 3])
            ```
        """
        return torch.cat(x, self.d)


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights="yolov5s.pt", device=torch.device("cpu"), dnn=False, data=None, fp16=False, fuse=True):
        """
        Initializes the DetectMultiBackend class for YOLOv5 with support for various inference backends.

        Supports multiple backends including PyTorch, TorchScript, ONNX Runtime, OpenVINO, CoreML, TensorRT,
        TensorFlow (SavedModel, GraphDef, Lite, Edge TPU), and PaddlePaddle. Initializes the model to the
        specified device and configures it for inference.

        Args:
            weights (str | list[str]): Path to the model weights file or a list of such paths. Supports various formats
                including '.pt' for PyTorch, '.onnx' for ONNX, '.engine' for TensorRT, etc.
            device (str): Device to run the inference on. Example: 'cpu', 'cuda:0'. Defaults to 'cpu'.
            dnn (bool): Use OpenCV DNN for ONNX inference. Default is False.
            data (str | None): Path to the dataset metadata file. Defaults to None if not provided.
            fp16 (bool): Enable FP16 precision. Default is False.
            fuse (bool): Fuse model layers. Default is True.

        Attributes:
            model (nn.Module | None): The initialized model ready for inference on the specified backend.
            stride (int): Stride of the model's convolutional layers.
            names (dict[int, str]): Dictionary mapping class IDs to names.
            ...

        Example:
            ```python
            from ultralytics import DetectMultiBackend
            model = DetectMultiBackend(weights="yolov5s.pt", device="cuda:0")
            ```

        Note:
            Supported backends:
            - PyTorch: weights = *.pt
            - TorchScript: *.torchscript
            - ONNX: *.onnx
            - OpenVINO: *_openvino_model
            - CoreML: *.mlmodel
            - TensorRT: *.engine
            - TensorFlow: *_saved_model, *.pb, *.tflite, *_edgetpu.tflite
            - PaddlePaddle: *_paddle_model
        """
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        fp16 &= pt or jit or onnx or engine or triton  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA
        if not (pt or triton):
            w = attempt_download(w)  # download if not local

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            LOGGER.info(f"Loading {w} for TorchScript inference...")
            extra_files = {"config.txt": ""}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files["config.txt"]:  # load metadata dict
                d = json.loads(
                    extra_files["config.txt"],
                    object_hook=lambda d: {int(k) if k.isdigit() else k: v for k, v in d.items()},
                )
                stride, names = int(d["stride"]), d["names"]
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f"Loading {w} for ONNX OpenCV DNN inference...")
            check_requirements("opencv-python>=4.5.4")
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f"Loading {w} for ONNX Runtime inference...")
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
            import onnxruntime

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if "stride" in meta:
                stride, names = int(meta["stride"]), eval(meta["names"])
        elif xml:  # OpenVINO
            LOGGER.info(f"Loading {w} for OpenVINO inference...")
            check_requirements("openvino>=2023.0")  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core, Layout, get_batch

            core = Core()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob("*.xml"))  # get *.xml file from *_openvino_model dir
            ov_model = core.read_model(model=w, weights=Path(w).with_suffix(".bin"))
            if ov_model.get_parameters()[0].get_layout().empty:
                ov_model.get_parameters()[0].set_layout(Layout("NCHW"))
            batch_dim = get_batch(ov_model)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            ov_compiled_model = core.compile_model(ov_model, device_name="AUTO")  # AUTO selects best available device
            stride, names = self._load_metadata(Path(w).with_suffix(".yaml"))  # load metadata
        elif engine:  # TensorRT
            LOGGER.info(f"Loading {w} for TensorRT inference...")
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download

            check_version(trt.__version__, "7.0.0", hard=True)  # require tensorrt>=7.0.0
            if device.type == "cpu":
                device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            is_trt10 = not hasattr(model, "num_bindings")
            num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
            for i in num:
                if is_trt10:
                    name = model.get_tensor_name(i)
                    dtype = trt.nptype(model.get_tensor_dtype(name))
                    is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                    if is_input:
                        if -1 in tuple(model.get_tensor_shape(name)):  # dynamic
                            dynamic = True
                            context.set_input_shape(name, tuple(model.get_profile_shape(name, 0)[2]))
                        if dtype == np.float16:
                            fp16 = True
                    else:  # output
                        output_names.append(name)
                    shape = tuple(context.get_tensor_shape(name))
                else:
                    name = model.get_binding_name(i)
                    dtype = trt.nptype(model.get_binding_dtype(i))
                    if model.binding_is_input(i):
                        if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                            dynamic = True
                            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                        if dtype == np.float16:
                            fp16 = True
                    else:  # output
                        output_names.append(name)
                    shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size
        elif coreml:  # CoreML
            LOGGER.info(f"Loading {w} for CoreML inference...")
            import coremltools as ct

            model = ct.models.MLModel(w)
        elif saved_model:  # TF SavedModel
            LOGGER.info(f"Loading {w} for TensorFlow SavedModel inference...")
            import tensorflow as tf

            keras = False  # assume TF1 saved_model
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
        elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            LOGGER.info(f"Loading {w} for TensorFlow GraphDef inference...")
            import tensorflow as tf

            def wrap_frozen_graph(gd, inputs, outputs):
                """Wraps a TensorFlow GraphDef for inference, returning a pruned function."""
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            def gd_outputs(gd):
                """Generates a sorted list of graph outputs excluding NoOp nodes and inputs, formatted as '<name>:0'."""
                name_list, input_list = [], []
                for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
                    name_list.append(node.name)
                    input_list.extend(node.input)
                return sorted(f"{x}:0" for x in list(set(name_list) - set(input_list)) if not x.startswith("NoOp"))

            gd = tf.Graph().as_graph_def()  # TF GraphDef
            with open(w, "rb") as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf

                Interpreter, load_delegate = (
                    tf.lite.Interpreter,
                    tf.lite.experimental.load_delegate,
                )
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                LOGGER.info(f"Loading {w} for TensorFlow Lite Edge TPU inference...")
                delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                    platform.system()
                ]
                interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
            else:  # TFLite
                LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
                interpreter = Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # load metadata
            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(w, "r") as model:
                    meta_file = model.namelist()[0]
                    meta = ast.literal_eval(model.read(meta_file).decode("utf-8"))
                    stride, names = int(meta["stride"]), meta["names"]
        elif tfjs:  # TF.js
            raise NotImplementedError("ERROR: YOLOv5 TF.js inference is not supported")
        elif paddle:  # PaddlePaddle
            LOGGER.info(f"Loading {w} for PaddlePaddle inference...")
            check_requirements("paddlepaddle-gpu" if cuda else "paddlepaddle")
            import paddle.inference as pdi

            if not Path(w).is_file():  # if not *.pdmodel
                w = next(Path(w).rglob("*.pdmodel"))  # get *.pdmodel file from *_paddle_model dir
            weights = Path(w).with_suffix(".pdiparams")
            config = pdi.Config(str(w), str(weights))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()
        elif triton:  # NVIDIA Triton Inference Server
            LOGGER.info(f"Using {w} as Triton Inference Server...")
            check_requirements("tritonclient[all]")
            from utils.triton import TritonRemoteModel

            model = TritonRemoteModel(url=w)
            nhwc = model.runtime.startswith("tensorflow")
        else:
            raise NotImplementedError(f"ERROR: {w} is not a supported format")

        # class names
        if "names" not in locals():
            names = yaml_load(data)["names"] if data else {i: f"class{i}" for i in range(999)}
        if names[0] == "n01440764" and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / "data/ImageNet.yaml")["names"]  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        """
        Return self.from_numpy(y) if isinstance(y, (np.ndarray)) else y.

        Args:
            im (torch.Tensor): Input image tensor of shape (N, C, H, W) where N is batch size, C is number of channels,
                H is height, and W is width.
            augment (bool): Flag to enable test-time augmentation. Defaults to False.
            visualize (bool): Flag to enable feature visualization. Defaults to False.

        Returns:
            (torch.Tensor | list[torch.Tensor] | np.ndarray): Model predictions, which could be a single torch.Tensor,
                a list of torch.Tensors, or a numpy array, mainly determined by the specific backend and its output format.

        Example:
            ```python
            # Example usage
            model = DetectMultiBackend(weights='yolov5s.pt')
            input_image = torch.rand(1, 3, 640, 640)  # Example image tensor
            predictions = model.forward(input_image)
            ```

        Note:
            This function supports a variety of backend models, such as torchscript, ONNX, OpenVINO, TensorRT, and more.
            Ensure the appropriate backend libraries are installed and the model is in the correct format for the chosen backend.
        """
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
        elif self.jit:  # TorchScript
            y = self.model(im)
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            y = list(self.ov_compiled_model(im).values())
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings["images"].shape:
                i = self.model.get_binding_index("images")
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings["images"].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs["images"] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        elif self.coreml:  # CoreML
            im = im.cpu().numpy()
            im = Image.fromarray((im[0] * 255).astype("uint8"))
            # im = im.resize((192, 320), Image.BILINEAR)
            y = self.model.predict({"image": im})  # coordinates are xywh normalized
            if "confidence" in y:
                box = xywh2xyxy(y["coordinates"] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y["confidence"].max(1), y["confidence"].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                y = list(reversed(y.values()))  # reversed for segmentation models (pred, proto)
        elif self.paddle:  # PaddlePaddle
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
        elif self.triton:  # NVIDIA Triton Inference Server
            y = self.model(im)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite or Edge TPU
                input = self.input_details[0]
                int8 = input["dtype"] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input["quantization"]
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input["index"], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output["index"])
                    if int8:
                        scale, zero_point = output["quantization"]
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    y.append(x)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """
        Loads model weights from a NumPy array.

        Args:
            x (np.ndarray): NumPy array containing the model weights to be loaded. The array should have the appropriate
                dimensions expected by the model.

        Returns:
            (torch.Tensor): PyTorch tensor containing the same values as the input NumPy array, transferred to the appropriate
                device being used by the model.

        Example:
            ```python
            import numpy as np
            weights = np.random.rand(1, 3, 640, 640)  # Example weight array
            model = DetectMultiBackend()  # Initialize DetectMultiBackend
            torch_weights = model.from_numpy(weights)  # Load weights as PyTorch tensor
            ```

        Note:
            This function ensures that the loaded weights are compatible with the device being used by the model, facilitating
            seamless conversion from NumPy arrays to PyTorch tensors.
        """
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        Perform a single forward pass to initialize model weights and optimize runtime performance.

        Args:
            imgsz (tuple[int]): Input image size in the form of (N, C, H, W), where N is the batch size, C is the number of
                channels, H is the height, and W is the width of the input tensor. Default is (1, 3, 640, 640).

        Returns:
            None

        Note:
            This function runs a forward pass with a dummy tensor to initialize the model and optimize runtime performance,
            especially useful when using devices like GPUs or Triton Inference Server.

        Example:
            ```python
            from ultralytics.models.common import DetectMultiBackend

            # Initialize the DetectMultiBackend model
            model = DetectMultiBackend(weights='yolov5s.pt', device=torch.device('cuda:0'))

            # Warmup the model with a dummy input
            model.warmup(imgsz=(1, 3, 640, 640))
            ```
        """
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p="path/to/model.pt"):
        """
        Determine the model type from the provided file path or URL, supporting various model export formats.

        Args:
            p (str): The file path or URL to the model (e.g., '/path/to/model.onnx').

        Returns:
            tuple[bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool]: A 13-element tuple with each
                boolean indicating the presence of one of the supported model types in the order (pt, jit, onnx, xml, engine,
                coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton).

        Notes:
            Supported model formats include:
            - PyTorch (.pt)
            - TorchScript (.torchscript)
            - ONNX (.onnx)
            - OpenVINO (.xml)
            - TensorRT (.engine)
            - CoreML (.mlmodel)
            - TensorFlow SavedModel (directory)
            - TensorFlow GraphDef (.pb)
            - TensorFlow Lite (.tflite)
            - TensorFlow Edge TPU (.edgetpu.tflite)
            - TensorFlow.js (tfjs)
            - PaddlePaddle (directory)
            - NVIDIA Triton Inference Server (http/grpc URL)

        Example:
            ```python
            model_type = DetectMultiBackend._model_type("/path/to/model.onnx")
            print(model_type)  # Output: (False, False, True, False, False, False, False, False, False, False, False, False, False)
            ```

        Related:
            Refer to the [Ultralytics YOLO documentation](https://github.com/ultralytics/yolov5) for additional
            information on supported backends.
        """
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from export import export_formats
        from utils.downloads import is_url

        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
        return types + [triton]

    @staticmethod
    def _load_metadata(f=Path("path/to/meta.yaml")):
        """
        Loads metadata from a YAML file and returns important details including stride and class names.

        Args:
            f (Path, optional): Path to the metadata YAML file. Defaults to "path/to/meta.yaml".

        Returns:
            tuple (int, list):
                int: The stride value for the model.
                list: The list of class names inferred from the metadata file.

        Example:
            ```python
            from pathlib import Path
            stride, names = DetectMultiBackend._load_metadata(Path('path/to/meta.yaml'))
            ```
        """
        if f.exists():
            d = yaml_load(f)
            return d["stride"], d["names"]  # assign stride, names
        return None, None


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model, verbose=True):
        """
        Initializes the AutoShape wrapper for YOLOv5 model, handling preprocessing, inference, and post-processing.

        Args:
            model (torch.nn.Module): The YOLOv5 model to wrap, which will be prepared for evaluation mode.
            verbose (bool): Whether to log information during initialization. Defaults to True.

        Returns:
            None

        Example:
            ```python
            from ultralytics import YOLOv5
            model = YOLOv5('yolov5s.pt')  # Load a YOLOv5 model
            model = AutoShape(model)  # Add AutoShape wrapper
            input_image = 'path/to/image.jpg'
            results = model(input_image)  # Perform model inference
            ```
        """
        super().__init__()
        if verbose:
            LOGGER.info("Adding AutoShape... ")
        copy_attr(self, model, include=("yaml", "nc", "hyp", "names", "stride", "abc"), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.inplace = False  # Detect.inplace=False for safe multithread inference
            m.export = True  # do not output loss values

    def _apply(self, fn):
        """
        Apply embedding in real-time tokenizer as embeddings initialized as empty.

        Args:
            embedding (torch.Tensor): The embedding tensor of tokens to be applied.
            normalize (bool, optional): Whether to normalize the embeddings during initialization. Default is False.

        Returns:
            (torch.FloatTensor): A tensor containing initialized token embeddings.

        Example:
            ```python
            tokenizer = RealTimeTokenizer()
            tokenizer.initialize_embedding(embedding=torch.rand(1000, 300))
            ```

        Note:
            Embedding tensor weights should be applied as per requirements in the RealTimeTokenizer for operational efficiency.
        """
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @smart_inference_mode()
    def forward(self, ims, size=640, augment=False, profile=False):
        """
        Perform inference on various input formats using the YOLO model with optional augmentations and profiling.

        Args:
            ims (str | Path | list | tuple | np.ndarray | torch.Tensor): Input images for inference; accepted formats include:
                - File path or URI: `ims = 'data/images/zidane.jpg'`
                - OpenCV image: `ims = cv2.imread('image.jpg')[:,:,::-1]`
                - PIL image: `ims = Image.open('image.jpg')`
                - Numpy array: `ims = np.zeros((640, 1280, 3))`
                - Torch tensor: `ims = torch.zeros(16, 3, 320, 640)`
                - Multiple images: `ims = [Image.open('image1.jpg'), Image.open('image2.jpg')]`
            size (int | tuple): Inference image size as (height, width), a single int will be expanded to (size, size).
            augment (bool): Apply test-time augmentations during inference.
            profile (bool): Enable/disable profiling for performance evaluation.

        Returns:
            (list[torch.Tensor]): List of detections for each input image. Each tensor is of shape (N, 6) where N is the number
                of detections, and columns represent (x1, y1, x2, y2, confidence, class).

        Example:
            ```python
            from PIL import Image
            images = [Image.open('image1.jpg'), Image.open('image2.jpg')]
            model = Autoshape(my_yolo_model)
            results = model.forward(images, size=640)
            for res in results:
                print(res)
            ```

        Note:
            This method supports Automatic Mixed Precision (AMP) for inference if enabled. If using AMP, ensure the input
            device is not set to "cpu".
        """
        # For size(height=640, width=1280), RGB images example inputs are:
        #   file:        ims = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        dt = (Profile(), Profile(), Profile())
        with dt[0]:
            if isinstance(size, int):  # expand
                size = (size, size)
            p = next(self.model.parameters()) if self.pt else torch.empty(1, device=self.model.device)  # param
            autocast = self.amp and (p.device.type != "cpu")  # Automatic Mixed Precision (AMP) inference
            if isinstance(ims, torch.Tensor):  # torch
                with amp.autocast(autocast):
                    return self.model(ims.to(p.device).type_as(p), augment=augment)  # inference

            # Pre-process
            n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])  # number, list of images
            shape0, shape1, files = [], [], []  # image and inference shapes, filenames
            for i, im in enumerate(ims):
                f = f"image{i}"  # filename
                if isinstance(im, (str, Path)):  # filename or uri
                    im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith("http") else im), im
                    im = np.asarray(exif_transpose(im))
                elif isinstance(im, Image.Image):  # PIL Image
                    im, f = np.asarray(exif_transpose(im)), getattr(im, "filename", f) or f
                files.append(Path(f).with_suffix(".jpg").name)
                if im.shape[0] < 5:  # image in CHW
                    im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
                im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # enforce 3ch input
                s = im.shape[:2]  # HWC
                shape0.append(s)  # image shape
                g = max(size) / max(s)  # gain
                shape1.append([int(y * g) for y in s])
                ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
            shape1 = [make_divisible(x, self.stride) for x in np.array(shape1).max(0)]  # inf shape
            x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # pad
            x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
            x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32

        with amp.autocast(autocast):
            # Inference
            with dt[1]:
                y = self.model(x, augment=augment)  # forward

            # Post-process
            with dt[2]:
                y = non_max_suppression(
                    y if self.dmb else y[0],
                    self.conf,
                    self.iou,
                    self.classes,
                    self.agnostic,
                    self.multi_label,
                    max_det=self.max_det,
                )  # NMS
                for i in range(n):
                    scale_boxes(shape1, y[i][:, :4], shape0[i])

            return Detections(ims, y, files, dt, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None):
        """
        Initialize a Detections instance, holding images, predictions, and metadata.

        Args:
            ims (list[np.ndarray]): List of input images in numpy array format.
            pred (list[torch.Tensor]): List of predicted tensors, each of shape (N, 6) containing [x1, y1, x2, y2, confidence, class].
            files (list[str]): List of filenames corresponding to input images.
            times (tuple[float, float, float], optional): Profiling times for various stages in milliseconds. Default is (0, 0, 0).
            names (list[str], optional): Class names corresponding to model's output classes. Default is `None`.
            shape (tuple[int, int], optional): Expected shape of the input tensor (C, H, W). Default is `None`.

        Returns:
            None

        Example:
            ```python
            # Example to initialize Detections class
            from ultralytics.models.common import Detections
            images = [np.random.rand(640, 480, 3) for _ in range(4)]  # Four random images
            preds = [torch.rand(10, 6) for _ in range(4)]  # Random predictions for each image
            filenames = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg']
            detection_instance = Detections(images, preds, filenames)
            ```

        Note:
            This class is typically used as a container for results after performing inference on a batch of images with a YOLOv5
            model. The predictions are assumed to be in the form of bounding boxes with associated confidence scores and class labels.
        """
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]  # normalizations
        self.ims = ims  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple(x.t / self.n * 1e3 for x in times)  # timestamps (ms)
        self.s = tuple(shape)  # inference BCHW shape

    def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path("")):
        """
        Performs detection post-processing tasks.

        Args:
            pprint (bool): If True, returns a formatted string with detection info for each image. Defaults to False.
            show (bool): If True, displays the images with bounding box annotations. Defaults to False.
            save (bool): If True, saves the annotated images to the specified directory. Defaults to False.
            crop (bool): If True, crops detected objects and optionally saves them. Defaults to False.
            render (bool): If True, modifies the self.ims with annotated images. Defaults to False.
            labels (bool): If True, adds labels to the bounding boxes in the annotations. Defaults to True.
            save_dir (Path): Directory to save annotated or cropped images. Defaults to current directory (Path("")).

        Returns:
            None or str: If pprint is True, returns a formatted string with detection information. Otherwise, returns None.

        Example:
            ```python
            # Create a Detections object
            detections = Detections(ims=images, pred=predictions, files=['image1.jpg'])

            # Run detection post-processing with saving and displaying images
            detections._run(pprint=True, show=True, save=True, save_dir=Path("/output"), crop=True)
            ```

        Note:
            This function processes the detection results, applying bounding boxes and labels to the images, and optionally
            displays or saves them. It also supports cropping detected objects and saving them separately.
        """
        s, crops = "", []
        for i, (im, pred) in enumerate(zip(self.ims, self.pred)):
            s += f"\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} "  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                s = s.rstrip(", ")
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f"{self.names[int(cls)]} {conf:.2f}"
                        if crop:
                            file = save_dir / "crops" / self.names[int(cls)] / self.files[i] if save else None
                            crops.append(
                                {
                                    "box": box,
                                    "conf": conf,
                                    "cls": cls,
                                    "label": label,
                                    "im": save_one_box(box, im, file=file, save=save),
                                }
                            )
                        else:  # all others
                            annotator.box_label(box, label if labels else "", color=colors(cls))
                    im = annotator.im
            else:
                s += "(no detections)"

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if show:
                if is_jupyter():
                    from IPython.display import display

                    display(im)
                else:
                    im.show(self.files[i])
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.ims[i] = np.asarray(im)
        if pprint:
            s = s.lstrip("\n")
            return f"{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}" % self.t
        if crop:
            if save:
                LOGGER.info(f"Saved results to {save_dir}\n")
            return crops

    @TryExcept("Showing images is not supported in this environment")
    def show(self, labels=True):
        """
        Displays detection results with optional labels.

        Args:
            labels (bool): Whether to display class labels on detected objects. Default is True.

        Returns:
            None

        Example:
            ```python
            from ultralytics import YOLOv5
            model = YOLOv5("yolov5s.pt")
            detection = model("image.jpg")  # Perform model inference
            detection.show(labels=True)  # Show detection results with labels
            ```

        Notes:
            This function uses the `TryExcept` utility internally to handle any environment-specific exceptions related
            to displaying images, providing a more robust user experience.
        """
        self._run(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir="runs/detect/exp", exist_ok=False):
        """
        Save the detection results to a specified directory as images or crops.

        Args:
            save_dir (str | pathlib.Path): Directory where images or crops will be saved. Defaults to "runs/detect/exp".
            labels (bool): If True, include labels on the saved images. Defaults to True.
            crops (bool): If True, save cropped objects instead of full images. Defaults to False.
            exist_ok (bool): If False, create a new directory if save_dir already exists; else overwrite.

        Returns:
            None

        Example:
            ```python
            detections.save(save_dir='output/', labels=True, crops=False, exist_ok=True)
            ```

        Note:
            The save_dir must be a valid directory path. If exist_ok is False, the function generates a new directory by
            appending a counter to avoid overwriting existing results.
        """
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
        self._run(save=True, labels=labels, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir="runs/detect/exp", exist_ok=False):
        """
        Crops detected objects out of images and saves them to the specified directory.

        Args:
            save (bool): Whether to save the cropped images to disk. Default is True.
            save_dir (str | pathlib.Path): Directory path where the cropped images will be saved if `save` is True.
                Default is "runs/detect/exp".
            exist_ok (bool): When set to False, a new directory will be created automatically if `save_dir` already exists.
                Default is False.

        Returns:
            list[dict]: A list containing dictionaries with details about each cropped image. Each dictionary contains
                'box' (coordinates of the bounding box), 'conf' (confidence score), 'cls' (class label index), 'label'
                (class label as string), and 'im' (the cropped image array).

        Example:
            ```python
            from ultralytics import YOLOv5

            model = YOLOv5("yolov5s.pt")
            results = model("image.jpg")  # Perform detection
            crops = results.crop(save=True, save_dir="runs/crops")
            for crop in crops:
                print(crop['label'], crop['box'])
            ```

        Note:
            - This method is typically called using detected objects from an inference run.
            - Saving and directory incrementing are handled internally.
        """
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
        return self._run(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self, labels=True):
        """
        Render visual representations of detected objects onto the input image.

        Args:
            labels (bool): If True, class labels will be rendered on the bounding boxes (default is True).

        Returns:
            (None): This method modifies the input images in place, rendering bounding boxes and optionally labels.

        Example:
            ```python
            detections = model(image)  # Perform detection
            detections.render(labels=True)  # Render bounding boxes with labels
            ```
        """
        self._run(render=True, labels=labels)  # render results
        return self.ims

    def pandas(self):
        """
        Converts detection results into Pandas DataFrames.

        The returned DataFrames provide box coordinates in various formats, along with confidence scores, predicted class
        indices, and class names.

        Returns:
            (dict[str, pd.DataFrame]): Dictionary containing DataFrames with four different box formats:
                - 'xyxy': DataFrame with columns ('xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name')
                - 'xyxyn': DataFrame with columns ('xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name') with
                    normalized coordinates
                - 'xywh': DataFrame with columns ('xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name')
                - 'xywhn': DataFrame with columns ('xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name')
                    with normalized coordinates

        Example:
            ```python
            results = model(img)  # Perform inference
            df = results.pandas().xyxy[0]  # Get first image results in xyxy format
            print(df)
            ```

        Note:
            This function requires the Pandas library (`import pandas as pd`). Ensure Pandas is installed and available in
            your environment.
        """
        new = copy(self)  # return copy
        ca = "xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"  # xyxy columns
        cb = "xcenter", "ycenter", "width", "height", "confidence", "class", "name"  # xywh columns
        for k, c in zip(["xyxy", "xyxyn", "xywh", "xywhn"], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        """
        Convert detection results into a list format for easier manipulation.

        Returns:
            (list[list[torch.Tensor]]): A list of detection results for each image, where each detection result is
                represented as a list of torch Tensors with varied shapes such as (N, M) for different formats like
                (xyxy, xywh).

        Example:
            ```python
            detections = Detections(ims, pred, files, times, names, shape)
            detection_list = detections.tolist()
            for det in detection_list:
                print(det)
            ```
        """
        r = range(self.n)  # iterable
        return [
            Detections(
                [self.ims[i]],
                [self.pred[i]],
                [self.files[i]],
                self.times,
                self.names,
                self.s,
            )
            for i in r
        ]

    def print(self):
        """
        Prints a summary of the detection results.

        Returns:
            None

        Example:
            ```python
            # Assume detections is an instance of Detections class obtained from model inference
            detections.print()
            ```
        """
        LOGGER.info(self.__str__())

    def __len__(self):
        """
        Returns:
            (int): The total number of detections.

        Example:
            ```python
            detections = model(image)  # Perform detection
            print(len(detections))  # Print the number of detections
            ```
        """
        return self.n

    def __str__(self):
        """
        Convert the Detections object to a readable string representation, detailing counts and detection performance.

        Returns:
            (str): Formatted string describing detections including inference timing and number of detections per image.

        Example:
            ```python
            detection_results = Detections(...)
            print(str(detection_results))
            # Output:
            # image 1/1: 640x640 3 persons, 1 bicycle, 3 cars
            # Speed: 10.0ms pre-process, 5.0ms inference, 1.0ms NMS per image at shape (1, 3, 640, 640)
            ```

        Note:
            This method provides a concise summary of the detection performance and the detected object counts per image in
            a batch.
        """
        return self._run(pprint=True)  # print results

    def __repr__(self):
        """
        Returns a formatted string representation of the Detections object.

        The string includes the number of detections per class and timing information for preprocessing, inference, and NMS
        (per image).

        Returns:
            (str): Formatted string detailing inference results, detection counts per class, and profiling times.

        Example:
            ```python
            detections = model('image.jpg')  # Perform inference
            print(detections.__repr__())
            ```

        Note:
            The string concatenates various details about the detections processed, making it suitable for logging or debugging.
        """
        return f"YOLOv5 {self.__class__} instance\n" + self.__str__()


class Proto(nn.Module):
    # YOLOv5 mask Proto module for segmentation models
    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes YOLOv5 Proto module for segmentation tasks, setting up convolutional layers and upsample operations.

        Args:
            c1 (int): Number of input channels to the Proto module.
            c_ (int): Number of intermediate channels for the hidden layers. Default is 256.
            c2 (int): Number of output channels for the segmentation masks. Default is 32.

        Example:
            ```python
            from ultralytics import YOLOv5, Proto
            model = YOLOv5()
            proto_layer = Proto(c1=256, c_=128, c2=64)
            ```

        Note:
            The Proto module is specifically designed for segmentation models within the YOLOv5 architecture. It includes
            convolutional and upsampling layers to process and upscale feature maps, generating masks for segmentation tasks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """
        Transform the input tensor through convolutional layers and upsampling.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W), where N is the batch size, C is the number of channels,
                and H and W are the height and width of the input tensor, respectively.

        Returns:
            (torch.Tensor): Output tensor with shape (N, C2, H*2, W*2), where C2 is the number of output channels defined
                during initialization.

        Example:
            ```python
            import torch
            from ultralytics.models.common import Proto

            proto_layer = Proto(c1=128, c_=256, c2=32)
            input_tensor = torch.rand(1, 128, 64, 64)
            output_tensor = proto_layer(input_tensor)
            print(output_tensor.shape)  # Output should be torch.Size([1, 32, 128, 128])
            ```

        Note:
            This module is designed for use in YOLOv5 segmentation models. It processes the input tensor through several
            convolutional layers and upsamples the output to twice its original height and width, with a configurable number of
            channels.
        """
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Classify(nn.Module):
    # YOLOv5 classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, dropout_p=0.0
    ):  # ch_in, ch_out, kernel, stride, padding, groups, dropout probability
        """
        Initializes the YOLOv5 classification head with a convolutional layer, adaptive average pooling, and dropout.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int, optional): Kernel size for the convolution layer. Defaults to 1.
            s (int, optional): Stride for the convolution layer. Defaults to 1.
            p (int | None, optional): Padding for the convolution layer. Defaults to None.
            g (int, optional): Number of groups for the convolution layer. Defaults to 1.
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.

        Returns:
            None

        Example:
            ```python
            classify_head = Classify(1280, 1000, k=1, s=1, dropout_p=0.5)
            ```
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=dropout_p, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Executes forward pass by applying convolution, pooling, dropout, and linear layers for classification."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
