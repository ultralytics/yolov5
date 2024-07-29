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
    Pads kernel to achieve 'same' output shape, taking into account optional dilation.

    Args:
        k (int | list[int]): Size of the kernel. Supports single integer or list of integers for each dimension.
        p (None | int | list[int]): Padding size. If None, computes 'same' padding automatically. Default is None.
        d (int): Dilation rate to apply to the kernel. Defaults to 1.

    Returns:
        (int | list[int]): Calculated padding size. Returns a single integer if the kernel size is an integer, otherwise a
            list of integers matching the dimensions of the kernel.

    Example:
        ```python
        pad_size = autopad(3)  # For a single dimension kernel of size 3, dilation 1
        pad_sizes = autopad([3, 3], d=2)  # For a 2D kernel with size 3x3 and dilation 2
        ```

    Note:
        This function is commonly used when creating neural network architectures to ensure the output dimensions
        match the input dimensions, facilitating easy model design and debugging.
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
        Initialize a convolutional layer with batch normalization and an optional activation function.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size. Default is 1.
            s (int): Stride size. Default is 1.
            p (None | int): Padding size. If None, the padding is computed as 'same' padding. Default is None.
            g (int): Number of groups for group convolution. Default is 1.
            d (int): Dilation rate. Default is 1.
            act (bool or torch.nn.Module): If True, uses the default activation function (SiLU), otherwise no activation
                is applied. You can also provide a custom activation function. Default is True.

        Returns:
            (None): This is an initialization method, so it does not return anything.

        Example:
            ```python
            # Creating a convolutional layer with 3 input channels, 16 output channels, kernel size 3, stride 1, and ReLU activation
            conv_layer = Conv(3, 16, k=3, s=1, act=torch.nn.ReLU())
            ```

        Note:
            The default activation function used is SiLU if `act` is set to True. You can replace it with other activation
            functions by passing the desired nn.Module as the `act` argument.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Perform convolution, batch normalization, and activation on the input tensor `x` in sequence.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C_in, H, W), where N is the batch size, C_in is the number of input
                channels, H is the height, and W is the width.

        Returns:
            (torch.Tensor): Output tensor after applying convolution, batch normalization, and activation, with shape
                (N, C_out, H_out, W_out) where C_out is the number of output channels and H_out, W_out are the heights and
                widths of the output based on the kernel size, stride, and padding.

        Example:
            ```python
            conv_layer = Conv(3, 16, k=3, s=1, p=1)
            input_tensor = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 input channels, 224x224 image
            output_tensor = conv_layer(input_tensor)
            ```

        Note:
            This forward pass integrates three operations: a convolution, batch normalization, and an optional activation function
            (default is nn.SiLU).
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Apply convolution and activation without batch normalization for optimized inference.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W), typically a feature map from previous layers.

        Returns:
            (torch.Tensor): Output tensor after applying convolution and activation, with shape (N, C', H', W') where the
                output channels C' may differ from input channels due to the convolution operations.

        Example:
            ```python
            conv_layer = Conv(3, 16, k=3, s=1, act=True)
            fused_output = conv_layer.forward_fuse(torch.rand(1, 3, 224, 224))
            ```
        """
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """
        Initializes a depth-wise convolution layer with optional activation.

        Args:
            c1 (int): Number of input channels (C1).
            c2 (int): Number of output channels (C2).
            k (int): Kernel size. Defaults to 1.
            s (int): Stride size. Defaults to 1.
            d (int): Dilation rate. Defaults to 1.
            act (bool | nn.Module): Activation function or flag. If True, SiLU activation is used. If a
                nn.Module is provided, it is used as the custom activation function. Defaults to True.

        Returns:
            None

        Example:
            ```python
            dwconv = DWConv(32, 64, 3, 1, 1, True)
            input_tensor = torch.rand(1, 32, 224, 224)  # Example input tensor with shape (N, C1, H, W)
            output_tensor = dwconv(input_tensor)  # Output tensor with shape (N, C2, H_out, W_out)
            ```
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """
        Initialize a depth-wise transpose convolutional layer.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size of the transpose convolution. Default is 1.
            s (int): Stride of the transpose convolution. Default is 1.
            p1 (int): Input padding for the transpose convolution. Default is 0.
            p2 (int): Output padding for the transpose convolution. Default is 0.

        Returns:
            None

        Example:
            ```python
            layer = DWConvTranspose2d(64, 128, 3, 2, 1, 1)
            output = layer(torch.randn(1, 64, 32, 32))  # Example input tensor
            ```
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        """
        Initialize a transformer layer without LayerNorm for improved performance.

        Args:
            c (int): Number of input and output channels for the transformer layer.
            num_heads (int): Number of attention heads in the multihead attention mechanism.

        Returns:
            None

        Example:
            ```python
            layer = TransformerLayer(c=512, num_heads=8)
            input_tensor = torch.rand(10, 32, 512)  # (sequence_length, batch_size, embedding_dim)
            output = layer(input_tensor)
            ```
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
        Perform forward pass with multihead attention and linear layers using residual connections.

        Args:
            x (torch.Tensor): Input tensor of shape (T, N, C) where T is the sequence length, N is the batch size,
                and C is the embedding dimension.

        Returns:
            (torch.Tensor): Output tensor of shape (T, N, C) matching the input shape.

        Example:
            ```python
            layer = TransformerLayer(c=512, num_heads=8)
            input_tensor = torch.rand(10, 32, 512)  # Example input tensor with shape (T, N, C)
            output = layer(input_tensor)  # Output tensor with same shape (T, N, C)
            ```

        Note:
            This implementation removes LayerNorm layers for better computational performance.
        """
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        """
        Initialize a Transformer block for vision tasks, adapting dimensions and stacking specified layers.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            num_heads (int): Number of attention heads in each transformer layer.
            num_layers (int): Number of transformer layers to stack.

        Returns:
            None

        Example:
            ```python
            transformer_block = TransformerBlock(c1=64, c2=128, num_heads=8, num_layers=6)
            ```

        Note:
            This implementation adapts to input dimension changes by including an initial convolution layer if required.
            Utilizes multi-head self-attention mechanism as described in the paper:
            https://arxiv.org/abs/2010.11929.
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
        Perform forward pass through the Vision Transformer block.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C1, W, H) where B is batch size, C1 is number of input channels,
                W is width and H is height.

        Returns:
            (torch.Tensor): Output tensor with shape (L, B, C2) after processing through Vision Transformer layers, where
                L is sequence length, B is batch size, and C2 is number of output channels.

        Example:
            ```python
            transformer_block = TransformerBlock(c1=3, c2=64, num_heads=8, num_layers=6)
            input_tensor = torch.rand(1, 3, 224, 224)  # Example input tensor of shape (B, C1, W, H)
            output_tensor = transformer_block(input_tensor)
            print(output_tensor.shape)  # Will output torch.Size([L, B, C2])
            ```

        Note:
            Ensure the input tensor has the correct shape (B, C1, W, H) and dimensions when using this Transformer block.
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
        Initialize a standard bottleneck layer.

        This layer consists of a sequence of convolution operations optionally followed by a shortcut connection. The bottleneck design helps in reducing the number of parameters while preserving the performance through embedding dimensionality reduction and restoration.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            shortcut (bool): Whether to add a shortcut connection. Defaults to True.
            g (int): Number of groups for group convolution. Defaults to 1.
            e (float): Expansion ratio for hidden layer dimensionality. Defaults to 0.5.

        Returns:
            (None): This function does not return any value.

        Example:
            ```python
            bottleneck_layer = Bottleneck(64, 128, shortcut=True, g=1, e=0.5)
            input_tensor = torch.randn(1, 64, 128, 128)  # Example input tensor with shape (N, C1, H, W)
            output_tensor = bottleneck_layer(input_tensor)  # Output tensor with shape (N, C2, H, W)
            ```

        Note:
            Ensure the input tensor to the Bottleneck layer has the correct shape (N, C1, H, W) where:
                N is the batch size,
                C1 is the number of input channels,
                H is the height,
                W is the width.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Perform a forward pass through the bottleneck layer with optional shortcut connection.

        Args:
            x (torch.Tensor): Input tensor with shape (..., C_in, H, W), where C_in is the number of input channels.

        Returns:
            (torch.Tensor): Output tensor, with shape (..., C_out, H, W) where C_out is the number of output channels, either including
                the shortcut connection if applicable.

        Example:
            ```python
            import torch
            from ultralytics.models.common import Bottleneck

            bottleneck = Bottleneck(c1=64, c2=64)
            x = torch.randn(1, 64, 144, 144)  # Sample input
            y = bottleneck(x)  # Forward pass
            ```
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize the CSP Bottleneck layer, which is an extension of the traditional bottleneck layer to leverage
        cross-stage partial connections.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of times the bottleneck layer is repeated. Default is 1.
            shortcut (bool): Whether to use shortcut connections in the bottleneck layers. Default is True.
            g (int): Number of groups for the grouped convolution in the bottleneck layers. Default is 1.
            e (float): Expansion factor to control the hidden channels in the bottleneck layers. Default is 0.5.

        Returns:
            (None): Initializes the parameters for the CSP Bottleneck module.

        Example:
            ```python
            from ultralytics.models.common import BottleneckCSP

            # Instantiate CSPBottleneck with specific configuration
            bottleneck_csp = BottleneckCSP(c1=64, c2=128, n=3, shortcut=True, g=2, e=0.5)

            # Example input tensor
            input_tensor = torch.randn(1, 64, 128, 128)  # Shape (B, C1, H, W)

            # Forward pass through the layer
            output_tensor = bottleneck_csp(input_tensor)
            print(output_tensor.shape)  # Should match expected output shape
            ```
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
        Perform a forward pass through the CSP (Cross Stage Partial) Bottleneck layer.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W) where N is the batch size, C is the number of channels,
                H is the height, and W is the width.

        Returns:
            (torch.Tensor): Output tensor after applying CSP bottleneck transformations, with shape (N, C2, H, W), where C2 is
                the output channel size specified during initialization.

        Example:
            ```python
            import torch
            from ultralytics.models.common import BottleneckCSP

            model = BottleneckCSP(c1=64, c2=128, n=1)
            x = torch.randn(1, 64, 128, 128)
            output = model.forward(x)
            ```

        Note:
            CSP Bottleneck architecture helps in reducing the amount of computation as well as mitigating the gradient
            vanishing problem in deep neural networks. The specific implementation follows the design principles outlined in
            https://github.com/WongKinYiu/CrossStagePartialNetworks.
        """
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        """
        Perform initialization of the CrossConv module, which combines convolutions with optional downsampling.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for the convolution. Defaults to 3.
            s (int): Stride for the convolution. Defaults to 1.
            g (int): Number of groups for the grouped convolution. Defaults to 1.
            e (float): Expansion factor for the intermediate channels. Defaults to 1.0.
            shortcut (bool): If True, includes a shortcut connection. Defaults to False.

        Returns:
            (None): This method initializes the CrossConv instance without returning any value.

        Note:
            This module is designed for channel expansion and downsampling operations within neural network architectures,
            particularly for YOLOv5.

        Example:
            ```python
            cross_conv = CrossConv(64, 128)
            input_tensor = torch.randn(1, 64, 224, 224)
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
        Perform feature downsampling, expansion, and optional shortcut connection in a neural network.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W) where N is the batch size, C is the number of channels,
                H is the height, and W is the width.

        Returns:
            (torch.Tensor): Output tensor with the same shape as the input, transformed through the cross convolution layers.

        Example:
            ```python
            import torch
            from ultralytics.models.common import CrossConv

            cross_conv = CrossConv(64, 128)
            input_tensor = torch.randn(1, 64, 224, 224)
            output_tensor = cross_conv(input_tensor)
            print(output_tensor.shape)  # Output tensor shape
            ```

        Note:
            CrossConv layers are used in models to effectively downsample and expand feature maps, aiding in feature extraction
            while maintaining computational efficiency.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize a CSP bottleneck containing three convolutional layers and optional shortcut connections.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of Bottleneck layers to include. Defaults to 1.
            shortcut (bool): Whether to use shortcut connections in the Bottleneck layers. Defaults to True.
            g (int): Number of groups for the grouped convolution. Defaults to 1.
            e (float): Expansion ratio for the hidden channels in the Bottleneck layers. Defaults to 0.5.

        Returns:
            (torch.Tensor): The output tensor from the sequential layers, maintaining the same spatial dimensions but potentially
            different channel dimensions.

        Example:
            ```python
            from ultralytics.models.common import C3
            import torch

            c3_layer = C3(c1=128, c2=256, n=1, shortcut=True)
            x = torch.randn(1, 128, 32, 32)  # Example input tensor
            y = c3_layer(x)  # Output tensor
            print(y.shape)  # torch.Size([1, 256, 32, 32])
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
        Performs a forward pass using CSP bottleneck with three convolution layers, incorporating hidden bottleneck
        layers.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W), where N is batch size, C is number of input channels,
                H is height, and W is width.

        Returns:
            (torch.Tensor): Output tensor with shape (N, C_out, H, W), where C_out is the number of output channels after
                processing through the CSP bottleneck with 3 convolutions.

        Example:
            ```python
            import torch
            from ultralytics.models.common import C3

            model = C3(c1=64, c2=128, n=3)
            input_tensor = torch.randn(1, 64, 128, 128)  # Example input tensor with shape (N, C, H, W)
            output_tensor = model(input_tensor)
            print(output_tensor.shape)  # Outputs tensor shape after forward pass
            ```

        Note:
            CSP Bottleneck with 3 convolutions and hidden bottleneck layers helps in efficient representation by downsampling
            and concatenating filtered features from different paths. This architecture is inspired by the principles outlined
            in https://github.com/WongKinYiu/CrossStagePartialNetworks.
        """
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize the C3x module with cross-convolutions.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of Bottleneck layers to include. Defaults to 1.
            shortcut (bool): Whether to add shortcut connections. Defaults to True.
            g (int): Number of groups for grouped convolution. Defaults to 1.
            e (float): Expansion ratio for hidden channels. Defaults to 0.5.

        Returns:
            None: This constructor initializes the C3x module with the specified parameters and does not return any value.

        Note:
            This class inherits from C3 and extends its functionality by adding cross-convolutions adjacent to the main
            bottleneck layers for enhanced feature extraction.

        Example:
            ```python
            c3x_layer = C3x(64, 128, n=3, shortcut=True, g=1, e=0.5)
            input_tensor = torch.randn(1, 64, 256, 256)  # Example input tensor with shape (N, C1, H, W)
            output_tensor = c3x_layer(input_tensor)  # Output tensor with shape (N, C2, H, W)
            ```
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize the C3 module with an integrated TransformerBlock for advanced feature extraction.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of Bottleneck layers to be stacked sequentially.
            shortcut (bool): Whether to use residual connections between layers.
            g (int): Number of groups in group convolution.
            e (float): Expansion coefficient for channel dimensions.

        Returns:
            None

        Example:
            ```python
            c3tr = C3TR(64, 128, n=3, shortcut=True, g=1, e=0.5)
            input_tensor = torch.rand(1, 64, 256, 256)  # Random input tensor with shape (B, C1, H, W)
            output_tensor = c3tr(input_tensor)
            ```

        Notes:
            This module extends C3 by incorporating a TransformerBlock for enhanced contextual feature extraction, as described in
            the paper "Attention Is All You Need" (https://arxiv.org/abs/2010.11929).
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize a C3 module with Spatial Pyramid Pooling (SPP) for advanced spatial feature extraction.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (tuple[int]): Kernel sizes for SPP. Defaults to (5, 9, 13).
            n (int, optional): Number of Bottleneck layers. Defaults to 1.
            shortcut (bool, optional): Whether to use residual connections. Defaults to True.
            g (int, optional): Number of groups for group convolution. Defaults to 1.
            e (float, optional): Expansion ratio for hidden channels. Defaults to 0.5.

        Returns:
            None

        Example:
            ```python
            c3spp = C3SPP(c1=64, c2=128, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5)
            input_tensor = torch.randn(1, 64, 32, 32)  # Batch size 1, 64 channels, 32x32 image
            output = c3spp(input_tensor)
            print(output.shape)  # Expected output shape: (1, 128, 32, 32)
            ```

        Note:
            The SPP layer enhances the receptive field size while keeping computational costs manageable.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initializes YOLOv5's C3 module using Ghost Bottlenecks for efficient feature extraction and processing.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int, optional): Number of Bottleneck layers to include. Defaults to 1.
            shortcut (bool, optional): Whether to add shortcut connections. Defaults to True.
            g (int, optional): Number of groups for group convolution. Defaults to 1.
            e (float, optional): Expansion ratio for hidden channels. Defaults to 0.5.

        Returns:
            None

        Example:
            ```python
            from ultralytics.models.common import C3Ghost

            c3ghost_layer = C3Ghost(c1=64, c2=128, n=2, shortcut=True, e=0.5)
            input_tensor = torch.randn(1, 64, 256, 256)  # Random input tensor with shape (B, C1, H, W)
            output_tensor = c3ghost_layer(input_tensor)
            ```
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        """
        Initialize the Spatial Pyramid Pooling (SPP) layer to enhance receptive field size and feature extraction.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (tuple[int], optional): Kernel sizes for max pooling layers. Default is (5, 9, 13).

        Returns:
            None

        Example:
            ```python
            spp_layer = SPP(c1=64, c2=128, k=(5, 9, 13))
            input_tensor = torch.randn(1, 64, 32, 32)  # Batch size 1, 64 channels, 32x32 resolution
            output_tensor = spp_layer(input_tensor)
            print(output_tensor.shape)  # Output shape: (1, 128, 32, 32)
            ```

        Note:
            The SPP layer facilitates effective extraction of multi-scale context by performing max pooling
            with multiple kernel sizes. This enhances the network's receptive field and robustness to object
            scaling and deformation.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """
        Apply the Spatial Pyramid Pooling (SPP) process to enhance spatial feature extraction from input tensor.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W), where N is batch size, C is the number of channels,
                H is the height, and W is the width.

        Returns:
            (torch.Tensor): Output tensor with enhanced spatial features, having shape (N, C2, H, W).

        Example:
            ```python
            spp_layer = SPP(c1=64, c2=128, k=(5, 9, 13))
            input_tensor = torch.randn(1, 64, 32, 32)  # Batch size 1, 64 channels, 32x32 spatial dimensions
            output_tensor = spp_layer(input_tensor)
            print(output_tensor.shape)  # Expected output shape: (1, 128, 32, 32)
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
        Initialize YOLOv5's Spatial Pyramid Pooling - Fast (SPPF) layer with convolution and max pooling.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int, optional): Kernel size for max pooling layers. Default is 5.

        Returns:
            None: This method initializes the SPPF layer without returning any value.

        Example:
            ```python
            sppf = SPPF(128, 256, k=5)
            input_tensor = torch.randn(1, 128, 64, 64)
            output_tensor = sppf(input_tensor)
            print(output_tensor.shape)  # Expected output shape: (1, 256, 64, 64)
            ```

        Note:
            SPPF enhances feature extraction efficiency by reducing spatial dimensions and enriching features using convolution and
            max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """
        Perform forward pass through the Spatial Pyramid Pooling-Fast (SPPF) layer to enhance spatial features.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C1, H, W), where N is batch size, C1 is number of input channels,
                H is height, and W is width.

        Returns:
            (torch.Tensor): Output tensor with enriched spatial features and shape (N, C2, H, W), where C2 is the number of
                output channels.

        Example:
            ```python
            sppf = SPPF(128, 256, k=5)
            input_tensor = torch.randn(1, 128, 64, 64)  # Example input tensor
            output_tensor = sppf(input_tensor)
            print(output_tensor.shape)  # Expected output shape: (1, 256, 64, 64)
            ```

        Note:
            The SPPF layer leverages multiple levels of max pooling to capture diverse spatial patterns efficiently, which is
            particularly useful in object detection tasks like YOLOv5.
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
        Initialize the Focus layer that concatenates slices of the input tensor to increase channel depth before
        applying convolution.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for the convolution. Defaults to 1.
            s (int): Stride for the convolution. Defaults to 1.
            p (int | None): Padding size for the convolution. Uses automatic padding if None. Defaults to None.
            g (int): Group size for the convolution. Defaults to 1.
            act (bool | nn.Module): Activation function to apply after the convolution. Uses default activation (nn.SiLU) if True,
                or no activation if False. Can also be a custom activation module.

        Returns:
            None: This is an initializer method, so it does not return a value.

        Example:
            ```python
            focus = Focus(3, 64, k=3, s=1, p=1)
            input_tensor = torch.rand(1, 3, 224, 224)
            output = focus(input_tensor)
            ```
        Notes:
            The Focus layer is designed to increase the channel dimension by concatenating four slices of the input tensor,
            then applying a convolution to the concatenated result.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Focus width and height information into channel space and apply convolution.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C1, H, W) where N is the batch size, C1 is the number of input
                channels, H is the height, and W is the width.

        Returns:
            (torch.Tensor): Output tensor with shape (N, C2, H/2, W/2) where C2 is the number of output channels specified
                during initialization.

        Example:
            ```python
            focus_layer = Focus(3, 64, k=3, s=1, p=1)
            input_tensor = torch.rand(1, 3, 224, 224)
            output_tensor = focus_layer(input_tensor)
            print(output_tensor.shape)  # Expected shape: (1, 64, 112, 112)
            ```

        Notes:
            The Focus layer increases the channel dimension by concatenating four slices of the input tensor, each slice being
            a downsampled version of the input. This effectively focuses width and height information into the channel space before
            applying convolution.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """
        Initialize a Ghost Convolution layer for efficient feature extraction using fewer parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int, optional): Kernel size for convolution. Default is 1.
            s (int, optional): Stride for convolution. Default is 1.
            g (int, optional): Number of groups for convolution, facilitating group-wise operations. Default is 1.
            act (bool | nn.Module, optional): Activation function to use. Default is True, which uses the default activation;
                can also accept an nn.Module for custom activation or False for no activation.

        Returns:
            (None): This method initializes the Ghost Convolution layer without returning any value.

        Example:
            ```python
            import torch
            from ultralytics.models.common import GhostConv

            x = torch.randn(1, 64, 128, 128)  # Example input tensor with shape (B, C1, H, W)
            conv_layer = GhostConv(64, 128)  # Initialize GhostConv with 64 input channels and 128 output channels
            y = conv_layer(x)  # Forward pass
            print(y.shape)  # Should output: torch.Size([1, 128, 128, 128])
            ```

        Note:
            The Ghost Convolution technique effectively reduces computational complexity by splitting convolution into two steps:
            a primary convolution and a series of cheaper operations to generate 'ghost' feature maps. The technique is published
            by Huawei Noah's Ark Lab and is aimed at optimizing neural network performance on edge devices.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """
        Perform a forward pass through the Ghost Convolution layer.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C1, H, W), where N is the batch size, C1 is the number of input
                channels, H is the height, and W is the width.

        Returns:
            (torch.Tensor): Output tensor with shape (N, C2, H, W), where C2 is the number of output channels after
                applying Ghost Convolution operations.

        Example:
            ```python
            import torch
            from ultralytics.models.common import GhostConv

            input_tensor = torch.randn(1, 64, 128, 128)  # Example input tensor with shape (B, C1, H, W)
            ghost_conv_layer = GhostConv(64, 128)  # Initialize GhostConv with 64 input channels and 128 output channels
            output_tensor = ghost_conv_layer.forward(input_tensor)  # Forward pass
            print(output_tensor.shape)  # Should output: torch.Size([1, 128, 128, 128])
            ```

        Note:
            Ghost Convolution aims to optimize feature extraction by combining standard convolutions with cheaper operations to
            generate 'ghost' feature maps, enhancing computational efficiency and performance.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):
        """
        Initialize a GhostBottleneck layer for efficient feature extraction and processing with optional downsampling.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depth-wise convolution. Defaults to 3.
            s (int): Stride for depth-wise convolution, determines downsampling. Defaults to 1.

        Returns:
            None

        Example:
            ```python
            from ultralytics.models.common import GhostBottleneck
            import torch

            # Initialize GhostBottleneck with 64 input channels, 128 output channels
            ghost_bottleneck = GhostBottleneck(c1=64, c2=128, k=3, s=2)
            x = torch.randn(1, 64, 56, 56)  # Example input tensor
            output = ghost_bottleneck(x)
            print(output.shape)  # Expected output tensor shape: (1, 128, 28, 28)
            ```

        Note:
            The GhostBottleneck module incorporates GhostConvs and optional depth-wise convolutions for efficient feature
            processing. The use of GhostConv layers reduces computational overhead while maintaining performance, making
            this bottleneck design suitable for deploying neural networks on resource-constrained devices. The specific
            implementation is inspired by the GhostNet architecture: https://github.com/huawei-noah/ghostnet.
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
        Performs a forward pass through the GhostBottleneck layer, leveraging Ghost convolution operations for efficient
        feature extraction.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W), where N is the batch size, C is the number of channels,
                H is the height, and W is the width.

        Returns:
            (torch.Tensor): Output tensor with shape (N, C2, H, W) after applying Ghost Convolutions and optional
                shortcut connections, where C2 is the number of output channels specified during initialization.

        Example:
            ```python
            import torch
            from ultralytics.models.common import GhostBottleneck

            # Initialize GhostBottleneck with 64 input and 128 output channels
            ghost_bottleneck = GhostBottleneck(64, 128)
            x = torch.randn(1, 64, 56, 56)  # Example input
            y = ghost_bottleneck(x)  # Forward pass
            print(y.shape)  # Output shape should be (1, 128, 56, 56)
            ```

        Note:
            This layer is part of the GhostNet architecture, designed for lightweight and efficient neural network models,
            particularly on edge devices. The architecture minimizes computational complexity by generating fewer primary
            feature maps and using cheap operations to produce 'ghost' feature maps.
        """
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        """
        Initialize the Contract module for transforming spatial dimensions into channels.

        Args:
            gain (int): The factor by which to contract the dimensions. For example, a gain of 2 will halve the spatial
                dimensions and quadruple the channel dimension.

        Example:
            ```python
            contract_layer = Contract(gain=2)
            x = torch.randn(1, 64, 80, 80)
            output = contract_layer(x)  # results in shape (1, 256, 40, 40)
            ```
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """
        Forward pass for contracting the spatial dimensions into the channel dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) where B is the batch size, C is the number of channels,
                H is the height, and W is the width.

        Returns:
            (torch.Tensor): Tensor with the spatial dimensions contracted into the channel dimension, with shape
                (B, C * gain * gain, H // gain, W // gain).

        Example:
            ```python
            contract_layer = Contract(gain=2)
            input_tensor = torch.randn(1, 64, 80, 80)
            output_tensor = contract_layer(input_tensor)
            assert output_tensor.shape == (1, 256, 40, 40)
            ```
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
        Initialize the Expand module to increase spatial dimensions by redistributing channels.

        Args:
            gain (int): Factor to redistribute channels into spatial dimensions. Default is 2.

        Returns:
            None

        Example:
            ```python
            expand_layer = Expand(gain=2)
            input_tensor = torch.randn(1, 64, 80, 80)
            output_tensor = expand_layer(input_tensor)  # Output shape will be (1, 16, 160, 160)
            ```
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """
        Expand channels into spatial dimensions, i.e., transforms tensor shape (B, C, H, W) to (B, C/(gain^2), H*gain,
        W*gain).

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W), where B is the batch size, C is the number of channels,
                H is the height, and W is the width.

        Returns:
            (torch.Tensor): Output tensor with expanded spatial dimensions and reshaped channels. For example, an input tensor
                with shape (B, C, H, W) is transformed into (B, C/(gain^2), H*gain, W*gain), where gain is the expansion factor.

        Example:
            ```python
            expand_layer = Expand(gain=2)
            input_tensor = torch.rand(1, 64, 80, 80)
            output_tensor = expand_layer(input_tensor)
            print(output_tensor.shape)  # Expected output: torch.Size([1, 16, 160, 160])
            ```

        Note:
            Ensure that the number of input channels `C` is divisible by `gain^2` to avoid reshaping errors.
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
        Initializes a Concat module to concatenate tensors along a specified dimension.

        Args:
            dimension (int): Dimension along which to concatenate the input tensors. Default is 1.

        Returns:
            None: This method initializes the Concat module without returning any value.

        Example:
            ```python
            concat_layer = Concat(dimension=1)
            input_tensor1 = torch.randn(2, 3, 64, 64)
            input_tensor2 = torch.randn(2, 3, 64, 64)
            output_tensor = concat_layer([input_tensor1, input_tensor2])
            print(output_tensor.shape)  # Expected output shape: (2, 6, 64, 64)
            ```
        """
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """
        Concatenate a list of tensors along a specified dimension.

        Args:
            x (list[torch.Tensor]): A list of tensors to concatenate along the specified dimension. Each tensor
                must have the same shape except along the concatenation dimension.

        Returns:
            (torch.Tensor): The concatenated tensor along the specified dimension.

        Example:
            ```python
            import torch

            t1 = torch.randn(2, 3)
            t2 = torch.randn(2, 3)
            concat_module = Concat(dimension=0)
            result = concat_module([t1, t2])
            print(result.shape)  # Output shape will be (4, 3)
            ```

        Note:
            The concatenation dimension is specified during the initialization of the Concat module. Ensure that
            all tensors to be concatenated have matching shapes except along the dimension specified.
        """
        return torch.cat(x, self.d)


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights="yolov5s.pt", device=torch.device("cpu"), dnn=False, data=None, fp16=False, fuse=True):
        """
        Initialize the DetectMultiBackend class for inference on multiple backends such as PyTorch, ONNX, TensorRT, and
        more.

        Args:
            weights (str | list[str]): Path to the model weights. Multiple weights can be specified as a list for ensemble.
                Supported extensions include .pt, .onnx, .torchscript, .xml, .engine, .mlmodel, .pb, .tflite, and more.
            device (torch.device): The device to run the model on, e.g., torch.device('cpu'), torch.device('cuda:0').
                Default is torch.device('cpu').
            dnn (bool): Flag to use OpenCV DNN for ONNX models. Default is False.
            data (str | None): Path to the dataset configuration file containing class names. If None, default names will be used.
                Default is None.
            fp16 (bool): Flag to enable half-precision FP16 inference. Default is False.
            fuse (bool): Flag to fuse model convolutions for improved runtime efficiency. Default is True.

        Returns:
            None

        Example:
            ```python
            from ultralytics import DetectMultiBackend

            model = DetectMultiBackend(weights='yolov5s.pt', device=torch.device('cuda:0'))
            ```

        Note:
            - Successfully supports multiple backends such as PyTorch, ONNX, TensorRT, OpenCV DNN, PaddlePaddle, TensorFlow, and more.
            - Ensure that appropriate dependency packages for various backends are installed as required.
            - Utilizes efficient pre-initializations and backend-specific optimizations defined within the `__init__` method to support diverse methods of model loading and inference.
        """
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlpackage
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
        Performs inference on input images with support for multiple backends (PyTorch, ONNX, TensorRT, etc.).

        Args:
            im (torch.Tensor): Input tensor containing images, with shape (B, C, H, W) where B is batch size, C is number of
                channels, H is height, and W is width.
            augment (bool): Boolean flag to perform data augmentation during inference. Defaults to False.
            visualize (bool): Boolean flag to store or visualize the features/activations. Defaults to False.

        Returns:
            (torch.Tensor): Inference output tensor. Depending on the backend, this can be a single torch.Tensor or a list of
            torch.Tensors. Each tensor contains detection results such as bounding boxes and class scores.

        Example:
            ```python
            import torch
            from ultralytics.models.common import DetectMultiBackend

            # Initialize the model for a specific backend
            model = DetectMultiBackend(weights='yolov5s.pt', device=torch.device('cpu'))

            # Example input tensor of shape (B, C, H, W)
            input_tensor = torch.randn(1, 3, 640, 640)

            # Perform inference
            output_tensor = model.forward(input_tensor)
            ```

        Note:
            This function handles input preprocessing, model inference, and postprocessing. It supports multiple deep learning
            backends such as PyTorch, ONNX, TensorRT, TensorFlow, and more, with device compatibility checks and backend-specific
            operations.
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
        Convert NumPy array `x` to a torch tensor, maintaining device compatibility.

        Args:
            x (numpy.ndarray): Input array to convert to torch tensor, with any shape.

        Returns:
            (torch.Tensor): Converted torch tensor with the same data and shape as input array.

        Example:
            ```python
            import numpy as np
            input_array = np.random.randn(3, 224, 224)  # Example input array
            tensor = detect_multi_backend_instance.from_numpy(input_array)
            print(tensor.shape)  # Should output: torch.Size([3, 224, 224])
            ```

        Note:
            This function ensures that the resulting torch tensor retains the appropriate device (CPU or GPU) setting.
        """
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        Warms up the model by performing initial inference to prepare weights and memory allocations.

        Args:
            imgsz (tuple[int]): Input image size tuple in the format (B, C, H, W), where B is batch size, C is number of
                channels, H is height, and W is width for the warmup run. Defaults to (1, 3, 640, 640).

        Returns:
            None

        Example:
            ```python
            detect_backend = DetectMultiBackend(weights='yolov5s.pt')
            detect_backend.warmup(imgsz=(1, 3, 320, 320))
            ```

        Note:
            The warmup process involves passing a blank tensor through the model to ensure that weights are moved to the
            selected device, and memory is allocated properly. This is particularly useful for models running on GPU or with
            FP16 precision.
        """
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p="path/to/model.pt"):
        """
        Determine the model type from a given file path or URL.

        Args:
            p (str): File path or URL for the model.
                Supported formats include PyTorch, TorchScript, ONNX, OpenVINO, TensorRT, CoreML, TensorFlow, TFLite, and PaddlePaddle.

        Returns:
            (tuple[bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool]):
                A tuple of booleans representing the type of model inferred from the file path or URL. Each boolean indicates:
                - PyTorch (.pt)
                - TorchScript
                - ONNX (.onnx)
                - OpenVINO (.xml)
                - TensorRT (.engine)
                - CoreML (.mlmodel)
                - TensorFlow SavedModel
                - TensorFlow GraphDef (.pb)
                - TensorFlow Lite (.tflite)
                - TensorFlow Edge TPU (.tflite)
                - TensorFlow.js
                - PaddlePaddle

        Example:
            ```python
            model_type = DetectMultiBackend._model_type("model.onnx")
            assert model_type == (False, False, True, False, False, False, False, False, False, False, False, False)
            ```

        Note:
            This method relies on the file suffix and URL scheme to determine the type of model. Use this method to
            programmatically infer the model type, facilitating subsequent backend-specific operations.
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
        Load metadata from a specified YAML file.

        Args:
            f (Path): The path to the YAML file containing metadata.

        Returns:
            (int, dict): A tuple containing the following:
                - stride (int): The stride value extracted from the YAML file.
                - names (dict): A dictionary of class names mapped by their index.

        Example:
            ```python
            from pathlib import Path
            metadata_path = Path("path/to/meta.yaml")
            stride, names = DetectMultiBackend._load_metadata(metadata_path)
            print(f"Stride: {stride}")
            print(f"Class Names: {names}")
            ```

        Note:
            Ensure the YAML file at the specified path exists and contains 'stride' and 'names' keys for successful metadata extraction.
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
        Initializes an input-robust YOLO model with preprocessing, inference, and post-processing capabilities.

        Args:
            model (torch.nn.Module): The YOLO model to be wrapped.
            verbose (bool): If True, logs information about the initialization. Defaults to True.

        Returns:
            None

        Example:
            ```python
            from ultralytics import YOLO
            from ultralytics.models.common import AutoShape

            model = YOLO("yolov5s.pt")
            auto_shape_model = AutoShape(model)
            input_image = "path/to/image.jpg"
            predictions = auto_shape_model(input_image)
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
        Apply a function to model tensors excluding parameters or registered buffers.

        Args:
            fn (Callable): The function to apply to the tensors. Common choices include `to()`, `cpu()`, `cuda()`, `half()`, etc.

        Returns:
            (AutoShape): The current instance with the function applied.

        Note:
            This method is useful for moving all tensors to a specific device (e.g., GPU) or changing their data types.
        ```python
            self = super()._apply(fn)
            if self.pt:
                m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
                m.stride = fn(m.stride)
                m.grid = list(map(fn, m.grid))
                if isinstance(m.anchor_grid, list):
                    m.anchor_grid = list(map(fn, m.anchor_grid))
        ```
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
        Perform inference on given image inputs with support for various input formats.

        Args:
            ims (str | list[str] | pathlib.Path | list[pathlib.Path] | np.ndarray | list[np.ndarray] | torch.Tensor |
                list[torch.Tensor] | PIL.Image.Image | list[PIL.Image.Image]):
                Input images. Supported formats:
                    - File path as string ('data/images/zidane.jpg') or Path object.
                    - URL as string ('https://ultralytics.com/images/zidane.jpg').
                    - OpenCV image (cv2.imread()) with shape (H, W, 3).
                    - PIL image (Image.open()) with shape (H, W, 3).
                    - NumPy array with shape (H, W, 3) or (B, C, H, W).
                    - Torch tensor with shape (B, C, H, W).
                    - List of any of the above.
            size (int | tuple[int, int], optional): Target size for resizing input images, specified as an integer or
                a tuple (height, width). Defaults to 640.
            augment (bool, optional): If True, apply image augmentations during inference. Defaults to False.
            profile (bool, optional): If True, profile the inference process. Defaults to False.

        Returns:
            (list[torch.Tensor]): List of detection results, where each tensor has shape (N, 6) representing
                (x1, y1, x2, y2, conf, cls) for each detection.

        Example:
            ```python
            from PIL import Image
            from ultralytics import YOLO

            model = YOLO("yolov5s.pt")
            img = Image.open("path/to/image.jpg")
            results = model.autoshape.forward(img)
            ```

        Note:
            Inference can be performed with Automatic Mixed Precision (AMP) if `amp` attribute is set to `True` and
            the current hardware supports it.
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
        Initialize the Detections object, which stores prediction results from the YOLO model.

        Args:
            ims (list[np.ndarray]): A list of images as numpy arrays, where each array represents an image in HWC format.
            pred (list[torch.Tensor]): List of tensors containing the predicted bounding boxes and scores for each image.
                Each tensor has shape (N, 6) for (x1, y1, x2, y2, conf, cls).
            files (list[str]): List of filenames corresponding to images.
            times (tuple[float, float, float], optional): Profiling times, default is (0, 0, 0).
            names (list[str], optional): List of class names used for predictions, default is None.
            shape (tuple[int, int], optional): Shape of the input image, given as (height, width), default is None.

        Returns:
            None

        Example:
            ```python
            ims = [cv2.imread("image1.jpg"), cv2.imread("image2.jpg")]
            pred = [torch.tensor([[50, 50, 200, 200, 0.9, 1]]), torch.tensor([[30, 30, 150, 150, 0.8, 0]])]
            names = ["class0", "class1"]
            files = ["image1.jpg", "image2.jpg"]
            times = (0.1, 0.2, 0.3)
            detections = Detections(ims, pred, files, times, names)
            ```

        Note:
            This class simplifies tracing of predictions through different stages of the YOLOv5 inference pipeline, including
            preprocessing, model inference, and postprocessing. The Detections class maintains a convenient interface to
            access both raw and normalized bounding box coordinates and associated metadata.
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
        Perform desired post-processing actions (e.g., pretty-print results, show images, save outputs).

        Args:
            pprint (bool): If True, pretty-print the detection results.
            show (bool): If True, display the detection results using the default image viewer.
            save (bool): If True, save the detection results to the specified directory.
            crop (bool): If True, crop detected objects and save them.
            render (bool): If True, render annotated results onto the images.
            labels (bool): If True, add labels to the bounding boxes in the rendered images.
            save_dir (Path): Directory where the processed results will be saved. This is only used if `save` or `crop`
                is True.

        Returns:
            (str | None): Formatted string of the results if `pprint` is True, otherwise None.

        Example:
            ```python
            det = Detections(ims, pred, files, times=[0.1, 0.2, 0.3], names=["person", "bike"])
            result_str = det._run(pprint=True, show=True, save=False, crop=False, render=False, labels=True,
                                  save_dir=Path("./outputs"))
            print(result_str)  # Prints the formatted string of results.
            ```

        Note:
            Ensure that the `save_dir` exists when saving the results. The function handles different modes of result
            presentation, such as showing images using default viewers or displaying in Jupyter notebooks.
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
            labels (bool): If True, include class labels and confidence scores in the displayed results.

        Returns:
            None: The function does not return anything.

        Example:
            ```python
            detections = model(im)  # Perform inference
            detections.show(labels=True)  # Display results with labels
            ```

        Note:
            This function leverages the `Annotator` class to draw bounding boxes and labels on images and then displays
            them using either Jupyter notebook's display function or the default image viewer in other environments.
        """
        self._run(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir="runs/detect/exp", exist_ok=False):
        """
        Save detection results with optional labeling and directory creation.

        Args:
            labels (bool): Flag to include labels on the saved images. Defaults to True.
            save_dir (str | Path): Directory path where result images and optionally cropped images will be saved.
                Defaults to 'runs/detect/exp'.
            exist_ok (bool): Flag to allow existing directory content without creating a new directory.
                Defaults to False.

        Returns:
            None

        Example:
            ```python
            detections = Detections(ims, pred, files, names=names)
            detections.save(labels=True, save_dir='runs/detect/exp', exist_ok=False)
            ```

        Note:
            If `exist_ok` is False, the function will create a unique directory by incrementing the name to avoid conflicts
            with existing directories. If `save` is True, the images and crops will be saved in the specified `save_dir`.

            This function is particularly useful to persist detection results for future reference, further analysis, or
            debugging.
        """
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
        self._run(save=True, labels=labels, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir="runs/detect/exp", exist_ok=False):
        """
        Crop detected objects from the input images.

        Args:
            save (bool): Whether to save the cropped images to disk. Default is True.
            save_dir (str): Directory to save the cropped images. Default is 'runs/detect/exp'.
            exist_ok (bool): Whether to overwrite the existing directory if it exists. Default is False.

        Returns:
            (list[dict]): List of dictionaries, each containing information about a cropped image, with the keys:
                - 'box' (torch.Tensor): Bounding box of the crop with shape (4,).
                - 'conf' (torch.Tensor): Confidence score of the detection.
                - 'cls' (torch.Tensor): Class of the detected object.
                - 'label' (str): Label string with class name and confidence score.
                - 'im' (np.ndarray): Cropped image as a numpy array.

        Example:
            ```python
            detections = model.detect(images)
            crops = detections.crop(save=True, save_dir='runs/crops')
            for crop in crops:
                print(crop['label'], crop['im'].shape)
            ```

        Note:
            - If `save` is True, the cropped images will be saved in the specified `save_dir`, which will be incremented
              automatically if `exist_ok` is False and the directory already exists.
            - This function returns both the cropped images and their metadata, which can be useful for further analysis
              or display.
        """
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
        return self._run(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self, labels=True):
        """
        Render detection results on an image by drawing the predicted bounding boxes and labels.

        Args:
            imgs (np.ndarray | List[np.ndarray]): List of images as NumPy arrays on which detections were made.
            annotator (ultradatasets.utils.plotting.Annotator, optional): The annotator instance used for drawing bounding boxes
                and labels. Default is None.

        Returns:
            (np.ndarray | List[np.ndarray]): The image or list of images with rendered detections.

        Example:
            ```python
            from PIL import Image
            import requests
            import io
            import torch
            from ultralytics import YOLO

            # Load a sample image
            img_url = 'https://ultralytics.com/images/zidane.jpg'
            response = requests.get(img_url)
            img = Image.open(io.BytesIO(response.content))

            # Load the YOLO model
            model = YOLO('yolov5s.pt')

            # Perform inference
            results = model(auto_shape.forward([img]))

            # Render the detections on the image
            rendered_img = results.render()

            # Show the image
            rendered_img.show()
            ```

        Note:
            - Input images should be in RGB format.
            - The function supports rendering on multiple images if a list of images is provided.
            - This function is mainly used for visualization purposes in notebooks or GUI applications.
        """
        self._run(render=True, labels=labels)  # render results
        return self.ims

    def pandas(self):
        """
        Convert detections to pandas DataFrames for each box format.

        Args:
            None

        Returns:
            (dict): Dictionary of pandas DataFrames, one for each box format (xyxy, xyxyn, xywh, xywhn):
                - xyxy: DataFrame with columns ["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"].
                - xyxyn: DataFrame with columns ["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"] (normalized).
                - xywh: DataFrame with columns ["xcenter", "ycenter", "width", "height", "confidence", "class", "name"].
                - xywhn: DataFrame with columns ["xcenter", "ycenter", "width", "height", "confidence", "class", "name"]
                    (normalized).

        Example:
            ```python
            results = infer_image(image_path)
            dfs = results.pandas()
            print(dfs['xyxy'][0])  # print DataFrame for 'xyxy' format
            ```
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
        Convert detection results to a list of individual detection results.

        Returns:
            (list[Detections]): A list where each element is a `Detections` object for a single image, maintaining all
                relevant detection attributes.

        Example:
            ```python
            detections = model.detect(imgs)
            detections_list = detections.tolist()
            for detection in detections_list:
                print(detection.pandas().xyxy)
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
        Logs detection results for each image, including class names and detection counts per class.

        Example:
            ```python
            detections = model.predict(images)
            detections.print()
            ```
        ```python
        def print(self):
            print(self._run(pprint=True))  # print results
        ```
        """
        LOGGER.info(self.__str__())

    def __len__(self):
        """
        Returns:
            (int): Number of detections.
        """
        return self.n

    def __str__(self):
        """
        Returns a concise string representation of the detection results.

        Returns:
            (str): A string summarizing the image detection results, including number of detections per class and their respective
                confidences, along with processing time details.

        Example:
            ```python
            detections = Detections(ims, pred, files, times, names, shape)
            print(str(detections))
            ```

        Notes:
            Used primarily for logging and quick inspection of detection outputs.
        """
        return self._run(pprint=True)  # print results

    def __repr__(self):
        """
        Return a string representation of the Detections object including its class and formatted results.

        Returns:
            (str): A string representation of the Detections object, including class name and formatted results of the
                detection process.

        Example:
            ```python
            detections = Detections(ims, pred, files, times, names, shape)
            print(repr(detections))
            ```

        Note:
            This function is particularly useful for debugging and logging purposes, providing a clear, concise summary of
            the Detections object.
        """
        return f"YOLOv5 {self.__class__} instance\n" + self.__str__()


class Proto(nn.Module):
    # YOLOv5 mask Proto module for segmentation models
    def __init__(self, c1, c_=256, c2=32):
        """
        Initialize the YOLOv5 Proto module for segmentation models.

        Args:
            c1 (int): Number of input channels.
            c_ (int): Number of intermediate channels, default is 256.
            c2 (int): Number of output channels, default is 32.

        Example:
            ```python
            from ultralytics.models.common import Proto
            proto = Proto(c1=64, c_=256, c2=32)
            ```
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """
        Applies convolutional layers and upsampling to generate segmentation masks from input tensor `x`.

        Args:
            x (torch.Tensor): Input feature map tensor with shape (N, C1, H, W) where N is the batch size, C1 is the number of
                input channels, H is the height, and W is the width.

        Returns:
            (torch.Tensor): Output tensor with shape (N, C2, H_out, W_out), where C2 is the number of mask channels, and
                H_out and W_out are the height and width after upsampling.

        Example:
            ```python
            proto_layer = Proto(c1=512, c_=256, c2=32)
            input_tensor = torch.rand(1, 512, 64, 64)  # Example input tensor
            output_tensor = proto_layer(input_tensor)
            print(output_tensor.shape)  # Expected output shape: (1, 32, 128, 128)
            ```
        """
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Classify(nn.Module):
    # YOLOv5 classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, dropout_p=0.0
    ):  # ch_in, ch_out, kernel, stride, padding, groups, dropout probability
        """
        Initializes the Classify module for YOLOv5, transforming input feature maps to classification scores.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels corresponding to the number of classes.
            k (int): Convolutional kernel size. Defaults to 1.
            s (int): Convolutional stride size. Defaults to 1.
            p (int | None): Convolutional padding size. Defaults to None which implies automatic padding.
            g (int): Number of groups in convolutional layer. Defaults to 1.
            dropout_p (float): Dropout probability. Defaults to 0.0.

        Returns:
            (None): This method initializes the Classify instance without returning any value.

        Example:
            ```python
            classify_head = Classify(c1=2048, c2=1000, k=1, s=1, dropout_p=0.5)
            output = classify_head(input_tensor)  # where input_tensor is of shape (B, 2048, 20, 20)
            ```
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=dropout_p, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """
        Forward pass for the YOLOv5 classification head.

        This method takes an input tensor, applies convolution, pooling, and linear layers to produce classification scores.

        Args:
            x (torch.Tensor | list[torch.Tensor]): Input tensor or list of tensors with shape (..., C_in, H, W),
                where C_in is the number of input channels.

        Returns:
            (torch.Tensor): Output tensor with shape (B, C_out), where B is the batch size and C_out is the number of classes.

        Example:
            ```python
            classify_head = Classify(c1=2048, c2=1000, k=1, s=1, dropout_p=0.5)
            input_tensor = torch.rand(8, 2048, 20, 20)  # Example input tensor with shape (B, C_in, H, W)
            output_tensor = classify_head(input_tensor)
            print(output_tensor.shape)  # Should output: torch.Size([8, 1000])
            ```
        """
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
