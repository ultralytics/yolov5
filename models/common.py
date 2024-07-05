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
    Pads the kernel to produce a 'same' output shape, adjusting for optional dilation; returns the appropriate padding size.
    
    Args:
    k (int | list[int]): Kernel size, specified as a single integer or a list of integers.
    p (int | list[int], optional): Padding size, specified as a single integer or a list of integers. Defaults to None.
    d (int, optional): Dilation rate for the kernel. Defaults to 1.
    
    Returns:
    int | list[int]: The calculated padding size, returned as a single integer or a list of integers to match the input `k`.
    
    Examples:
    ```python
    # Example 1: Simple kernel with no dilation
    pad_size = autopad(k=3)  # Returns 1
    
    # Example 2: Kernel list with no dilation
    pad_size = autopad(k=[3, 5])  # Returns [1, 2]
    
    # Example 3: Kernel with dilation
    pad_size = autopad(k=3, d=2)  # Returns 2
    ```
    
    Note:
    This function is crucial for ensuring the output dimensions remain consistent when applying convolutions, particularly in convolutional neural network (CNN) operations.
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
        Initializes a standard convolution layer with optional batch normalization and activation.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int | tuple[int, int], optional): Size of the convolution kernel. Defaults to 1.
            s (int | tuple[int, int], optional): Stride of the convolution. Defaults to 1.
            p (int | tuple[int, int] | None, optional): Padding added to all four sides of the input. 
                Defaults to None, which applies automatic padding to achieve 'same' output shape.
            g (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
            d (int, optional): Dilation rate for the convolution. Defaults to 1.
            act (bool | nn.Module, optional): If True, applies default activation function (SiLU). If nn.Module, applies 
                the specified activation function. Defaults to True.
        
        Returns:
            Conv: A Conv object representing a convolutional layer followed by batch normalization and an optional 
            activation function.
        
        Examples:
            ```python
            import torch
            from ultralytics import Conv
        
            # Example instantiation
            conv_layer = Conv(3, 16, k=3, s=2, act=True)
            # Applying to an input tensor
            input_tensor = torch.randn(1, 3, 224, 224)
            output_tensor = conv_layer(input_tensor)
            ```
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Applies a convolution followed by batch normalization and an activation function to the input tensor `x`.
        
        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, channels, height, width)` on which the transformation will be applied.
        
        Returns:
            torch.Tensor: Transformed tensor of shape `(batch_size, out_channels, out_height, out_width)` after applying convolution,
                          batch normalization, and activation function.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Applies a fused convolution and activation function to the input tensor `x`.
        
        Args:
            x (torch.Tensor): Input tensor to be processed by the fused convolution and activation function.
        
        Returns:
            torch.Tensor: Output tensor after applying the fused convolution and activation.
        """
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """
        Initializes a depth-wise convolution layer with optional activation.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int, optional): Kernel size of the convolution. Default is 1.
            s (int, optional): Stride of the convolution. Default is 1.
            d (int, optional): Dilation rate of the convolution. Default is 1.
            act (bool | nn.Module, optional): Activation function or a flag to use the default activation 
                                               (nn.SiLU). If set to False, no activation is applied. Default is True.
        
        Returns:
            None
        
        Example:
            ```python
            dwconv = DWConv(3, 32, k=3, s=1, d=1, act=True)
            ```
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """
        Initializes a depth-wise transpose convolutional layer for YOLOv5.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int, optional): Kernel size. Defaults to 1.
            s (int, optional): Stride size. Defaults to 1.
            p1 (int, optional): Input padding. Defaults to 0.
            p2 (int, optional): Output padding. Defaults to 0.
        
        Returns:
            None
        
        Notes:
            The depth-wise transpose convolutional layer is primarily used for upsampling in the YOLOv5 architecture,
            enabling finer feature granularity in the generated feature maps.
        
        Examples:
            ```python
            conv_layer = DWConvTranspose2d(c1=64, c2=128, k=3, s=2, p1=1, p2=1)
            ```
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        """
        Initializes a transformer layer, sans LayerNorm for performance, with multihead attention and linear layers.
        
        Args:
            c (int): Number of channels or feature dimensions in the input tensor and the linear transformations.
            num_heads (int): Number of attention heads in the multihead attention mechanism.
        
        Notes:
            This transformer layer implementation omits LayerNorm layers for better performance, as described in:
            https://arxiv.org/abs/2010.11929
        
        Returns:
            None: This constructor does not return a value.
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
        Performs forward pass using MultiheadAttention followed by two linear transformations with residual connections.
        
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, embed_dim).
        
        Returns:
            torch.Tensor: Transformed tensor of shape (seq_len, batch_size, embed_dim).
        
        Notes:
            This function does not include LayerNorm layers for improved performance, as detailed in
            https://arxiv.org/abs/2010.11929.
        """
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        """
        Initializes a Transformer block for vision tasks, adapting input dimensions and stacking specified transformer layers.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            num_heads (int): Number of attention heads in MultiheadAttention.
            num_layers (int): Number of transformer layers to stack.
        
        Returns:
            None: This is an initializer method that sets up the Transformer block and does not return a value.
        
        Notes:
            If the number of input channels (`c1`) does not match the number of output channels (`c2`), a convolutional layer will be used to adapt the dimensions.
        
        Example:
            ```python
            transformer_block = TransformerBlock(c1=64, c2=128, num_heads=8, num_layers=6)
            ```
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
        Performs a forward pass through the Vision Transformer block including an optional convolution layer, a learnable 
        position embedding, and stacked Transformer layers.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, width, height).
        
        Returns:
            torch.Tensor: Output tensor after applying the Vision Transformer block. (torch.Tensor)
        
        Notes:
            Vision Transformer implementation as described in "An Image is Worth 16x16 Words: Transformers for Image 
            Recognition at Scale" (https://arxiv.org/abs/2010.11929).
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
        Initializes a standard bottleneck layer with optional shortcut connection and group convolution, supporting channel expansion.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            shortcut (bool): Whether to add a residual shortcut connection (default is True).
            g (int): Number of groups for group convolution (default is 1).
            e (float): Channel expansion factor (default is 0.5).
        
        Returns:
            None
        
        Note:
            This class utilizes two convolutional layers for bottleneck operations, with a potential shortcut connection
            for enhanced gradient flow and model performance. Within complex architectures like YOLOv5, bottlenecks help
            in reducing the number of parameters, thereby optimizing inference speed without compromising accuracy.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Processes the input tensor through two convolutional layers, with an optional shortcut connection.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W), where N is the batch size, C is the number of channels,
                              H is the height, and W is the width.
        
        Returns:
            torch.Tensor: Output tensor after applying the bottleneck operations, potentially with a residual/shortcut connection
                          (torch.Tensor).
        
        Example:
            ```python
            import torch
            from ultralytics import Bottleneck
        
            bottleneck = Bottleneck(64, 128)
            x = torch.randn(1, 64, 32, 32)
            output = bottleneck(x)
            ```
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initializes a CSP bottleneck module designed for better gradient flow, featuring optional shortcuts, group 
        convolutions, and channel expansion.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of bottleneck layers to repeat. Default is 1.
            shortcut (bool): Whether to add residual connections (shortcuts) within the block. Default is True.
            g (int): Number of groups for convolution layers. Default is 1.
            e (float): Expansion ratio to control the number of hidden channels. Default is 0.5.
        
        Returns:
            None
        
        Notes:
            This implementation follows the CSPNet architecture as specified in the 
            [Cross Stage Partial Networks (CSPNet) paper](https://github.com/WongKinYiu/CrossStagePartialNetworks).
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
        Processes input through the CSP bottleneck layers, applying convolutions, activations, and concatenation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width) to be processed.
        
        Returns:
            torch.Tensor: Output tensor after applying CSP bottleneck transformations, maintaining the 
            dimensionality of the input tensor.
        
        Example:
            ```python
            import torch
            from ultralytics.modules import BottleneckCSP
        
            x = torch.randn(1, 64, 128, 128)  # Example input tensor
            model = BottleneckCSP(c1=64, c2=128, n=3, shortcut=True, g=1, e=0.5)
            output = model(x)
            print(output.shape)  # Output tensor shape
            ```
        """
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        """
        Initializes CrossConv with downsampling, expanding, and optionally shortcutting; `c1` input, `c2` output channels.
        
        Args:
          c1 (int): Number of input channels.
          c2 (int): Number of output channels.
          k (int): Kernel size integer for convolutional layers. Default is 3.
          s (int): Stride size for convolutional layers. Default is 1.
          g (int): Number of groups for group convolution. Default is 1.
          e (float): Expansion factor to determine hidden channels. Default is 1.0.
          shortcut (bool): Flag indicating whether to add a shortcut connection. Default is False.
        
        Returns:
          None
        
        Example:
          ```python
          cross_conv = CrossConv(c1=64, c2=128, k=3, s=1, g=1, e=1.0, shortcut=True)
          ```
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Processes the input tensor with cross convolution operations, optionally adding shortcuts.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W), where N is the batch size, C is the number
                of channels, H is the height, and W is the width.
        
        Returns:
            torch.Tensor: Output tensor after applying cross convolutions and optional shortcut.
            
        Raises:
            ValueError: If the shape of `x` is invalid or dimensions do not match expected input size.
        
        Example:
            ```python
            import torch
            from ultralytics import CrossConv
        
            x = torch.randn(1, 32, 128, 128)  # Example input tensor
            model = CrossConv(32, 64, k=3, s=1, g=1, e=1.0, shortcut=False)
            output = model.forward(x)
            print(output.shape)  # Output tensor shape
            ```
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initializes a CSP-style bottleneck module with three convolutions, configurable shortcuts, and bottleneck repetition.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of bottleneck layers to repeat in the module. Default is 1.
            shortcut (bool): Whether to use shortcut connections. Default is True.
            g (int): Number of groups for group convolution inside the bottleneck layers. Default is 1.
            e (float): Expansion factor for bottleneck layers. Default is 0.5.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """
        Processes the input tensor `x` through multiple convolution layers and a sequence of bottleneck layers, applying feature
        extraction and combining results to produce an enhanced output.
        
        Args:
          x (torch.Tensor): Input tensor with shape (N, C, H, W), where N is batch size, C is number of channels, H and W are height and width.
        
        Returns:
          torch.Tensor: Output tensor post feature extraction and combination, with enhanced representation of input (same shape as input).
        
        Notes:
          This method applies convolution operations followed by bottleneck layers, concatenates their results, and passes 
          through a subsequent convolution to refine the features. This process is crucial for extracting rich feature maps 
          necessary for robust object detection and classification processes.
        """
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initializes C3x module with cross-convolutions, extending the C3 module with customizable channel dimensions, groups, 
        and expansion parameters.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of Bottleneck layers to be repeated. Default is 1.
            shortcut (bool): If True, enables shortcut connections. Default is True.
            g (int): Number of groups for group convolution. Default is 1.
            e (float): Expansion ratio for hidden channels. Default is 0.5.
        
        Returns:
            None
        
        Examples:
            ```python
            from ultralytics import C3x
            import torch
        
            # Example usage
            c3x_layer = C3x(c1=64, c2=128, n=4, shortcut=True, g=1, e=0.5)
            input_tensor = torch.rand(1, 64, 256, 256)  # batch size 1, 64 channels, 256x256 resolution
            output_tensor = c3x_layer(input_tensor)
            print(output_tensor.shape)  # should be [1, 128, 256, 256]
            ```
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        C3TR.__init__()
        
        Initializes C3 module with TransformerBlock for enhanced feature extraction, including options for channel sizes,
        shortcut configuration, group convolutions, and expansion factor.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of Bottleneck repetitions.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Number of groups for group convolutions.
            e (float): Expansion factor to increase the width of the hidden layers.
        
        Returns:
            None: This constructor does not return any value.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        """
        Initializes a C3 module integrated with an SPP layer, enabling enhanced spatial feature extraction.
        
        Args:
          c1 (int): Number of input channels.
          c2 (int): Number of output channels.
          k (tuple[int], optional): Tuple of kernel sizes for the SPP layer (default is (5, 9, 13)).
          n (int, optional): Number of Bottleneck layers to apply in the module (default is 1).
          shortcut (bool, optional): If True, add residual connections from input to output (default is True).
          g (int, optional): Number of groups for convolution layers (default is 1).
          e (float, optional): Expansion ratio for the hidden channels (default is 0.5).
        
        Returns:
          None
        
        Notes:
          This module combines the C3 structure with Spatial Pyramid Pooling (SPP) to capture multi-scale features,
          particularly useful for detecting objects at different scales.
        
        Examples:
        ```python
        from ultralytics import C3SPP
        
        # Initialize C3SPP module with 64 input channels, 128 output channels, and default settings
        c3_spp = C3SPP(c1=64, c2=128)
        ```
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initializes the YOLOv5 C3 module with Ghost Bottlenecks for efficient feature extraction.
        
        Args:
          c1 (int): Number of input channels.
          c2 (int): Number of output channels.
          n (int): Number of Bottleneck repetitions. Default is 1.
          shortcut (bool): Whether to use shortcuts within the bottlenecks. Default is True.
          g (int): Number of groups for the convolutions. Default is 1.
          e (float): Expansion ratio for the bottleneck layers. Default is 0.5.
        
        Returns:
          None: This constructor method initializes the module and does not return any value.
        
        Notes:
          The Ghost Bottleneck technique is used to provide enhanced feature extraction efficiency by reducing redundant computations.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        """
        SPP.__init__(self), initializes an SPP layer with Spatial Pyramid Pooling
        
        Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        k (Tuple[int, int, int]): Kernel sizes, defaults to (5, 9, 13).
        
        Returns:
        None
        
        Notes:
        This method initializes the layers used for Spatial Pyramid Pooling (SPP) as described in https://arxiv.org/abs/1406.4729, including convolutions and multi-scale pooling operations to enhance spatial features in the network.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """
        Provides forward propagation for the SPP (Spatial Pyramid Pooling) layer, utilizing multiple max-pooling layers.
        
        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W) where N is batch size, C is number of input 
                              channels, H is height, and W is width.
        
        Returns:
            torch.Tensor: Output tensor after applying a sequence of convolutions and max-pool operations, 
                          concatenated along the channel dimension.
            
        Example:
            ```python
            spp_layer = SPP(c1=256, c2=512, k=(5, 9, 13))
            input_tensor = torch.randn(1, 256, 64, 64)
            output_tensor = spp_layer(input_tensor)
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
        Initializes a Spatial Pyramid Pooling - Fast (SPPF) layer in YOLOv5 with specified input and output channels, and kernel size.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for max pooling layers. Defaults to 5.
        
        Returns:
            None: This constructor initializes the module and does not return a value.
        
        Notes:
            This layer is designed to accelerate the Spatial Pyramid Pooling (SPP) process by using a single fixed-size max pooling layer.
            SPPF is an efficient alternative to the standard SPP with kernel sizes (5, 9, 13).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """
        Processes input through a series of convolutional and max pooling layers to perform efficient spatial pyramid pooling,
        returning the enhanced feature tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) where N is the batch size, C is the number of channels, H is
                the height, and W is the width.
        
        Returns:
            torch.Tensor: Output tensor with enhanced feature maps after spatial pyramid pooling. The shape of this tensor
                retains the same height (H) and width (W) dimensions as the input tensor, but the channel dimension may be
                altered due to the convolutions.
        
        Example:
            ```python
            import torch
            from ultralytics.models.yolo import SPPF  # Ensure the path to SPPF is correct based on library structure
            
            sppf_layer = SPPF(c1=512, c2=1024, k=5)
            input_tensor = torch.randn(8, 512, 32, 32)  # Example input
            output = sppf_layer(input_tensor)
            print(output.shape)  # Expected output shape: (8, 1024, 32, 32)
            ```
        
        Notes:
            - This method suppresses PyTorch warnings related to max pooling with stride and padding during execution.
            - The SPPF layer is designed to provide a computationally efficient alternative to the standard SPP layer,
              preserving spatial information via multiple pooling operations.
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
        Initializes Focus module to concentrate spatial information into channel space via slice and concatenate followed by convolution.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int, optional): Convolution kernel size. Defaults to 1.
            s (int, optional): Convolution stride. Defaults to 1.
            p (int | None, optional): Convolution padding. Defaults to None (autopad).
            g (int, optional): Number of convolution groups. Defaults to 1.
            act (bool | nn.Module, optional): Activation function, if True uses default activation (SiLU), if False uses nn.Identity(), else accepts any nn.Module as activation function. Defaults to True.
        
        Returns:
            None
        
        Example:
            ```python
            focus_layer = Focus(c1=64, c2=128, k=3, s=2)
            output = focus_layer(input_tensor)
            ```
        Notes:
            The Focus layer is particularly useful in object detection models like YOLO, where spatial resolution is progressively reduced while channel depth is increased, extracting and concentrating feature-rich information.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Focuses width-height information into channel space and applies convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Transformed tensor of shape (batch_size, out_channels, height/2, width/2).
        
        Notes:
            This method splits the input tensor into four slices, each containing information from various
            quadrants of the height and width dimensions, and then concatenates these along the channel dimension,
            followed by a convolution operation.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """
        Initializes the Ghost Convolution layer with GhostNet's principles for efficient feature extraction.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size of the convolution. Default is 1.
            s (int): Stride size of the convolution. Default is 1.
            g (int): Number of blocked connections from input channels to output channels. Default is 1.
            act (bool | nn.Module): Activation function to use. If True, defaults to nn.SiLU(); if a PyTorch module is 
                provided, that will be used as the activation function; if False, no activation will be applied. Default is True.
        
        Returns:
            None
        
        Notes:
            This class implements the Ghost Convolution as described in the paper: 
            "GhostNet: More Features from Cheap Operations" (https://github.com/huawei-noah/ghostnet).
        
        Examples:
            ```python
            import torch
            from ultralytics import GhostConv
            
            # Example usage
            ghost_conv = GhostConv(c1=16, c2=32, k=3, s=1)
            x = torch.randn(1, 16, 64, 64)
            out = ghost_conv(x)
            ```
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """
        Performs forward pass, concatenating outputs of two convolutions on input tensor `x`.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W), where B is the batch size, C is the number 
                              of channels, H is the height, and W is the width.
        
        Returns:
            torch.Tensor: Output tensor after applying ghost convolution, maintaining the shape (B, C, H, W).
        
        Notes:
            This method is part of the Ghost Convolution module designed for efficient feature extraction. It 
            splits the input channels into two parts and applies standard convolution to one part while 
            applying a cheaper 'ghost' convolution to the other part before concatenating the results. The 
            GhostNet paper provides more details: https://github.com/huawei-noah/ghostnet.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):
        """
        Initializes a GhostBottleneck module for efficient feature extraction in neural networks.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution. Default is 3.
            s (int): Stride for depthwise convolution. Default is 1.
        
        Returns:
            None
        
        Notes:
            Ghost bottlenecks are designed to reduce computational cost and model size by generating feature maps via cheap operations.
            For more details, visit https://github.com/huawei-noah/ghostnet
        
        Examples:
        ```python
        ghost_bottleneck = GhostBottleneck(c1=64, c2=128, k=3, s=2)
        output = ghost_bottleneck(input_tensor)
        ```
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
        Processes the input tensor through a sequence of convolutional layers and an optional shortcut connection.
        
        Args:
            x (torch.Tensor): Input tensor with shape `(batch_size, channels, height, width)`.
        
        Returns:
            torch.Tensor: Output tensor after processing, maintaining the same batch size and spatial dimensions.
        """
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        """
        Initializes a layer to contract spatial dimensions (width-height) into channels with a specified gain factor.
        
        Args:
            gain (int): Factor by which width and height will be reduced while increasing channels. Default is 2.
        
        Returns:
            None: This method initializes the Contract layer without returning any value.
        
        Example:
            ```python
            from ultralytics.yolo import Contract
        
            # Create a Contract layer that reduces spatial dimensions by a factor of 2
            contraction_layer = Contract(gain=2)
            ```
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """
        Concentrates width and height dimensions into the channel dimension of the input tensor `x`.
        
        Args:
            x (torch.Tensor): Input tensor with shape `(batch_size, channels, height, width)`.
        
        Returns:
            torch.Tensor: Output tensor with expanded channel dimensions and reduced spatial dimensions.
            
        Example:
            ```python
            model = Contract(gain=2)
            x = torch.randn(1, 64, 40, 40)
            y = model.forward(x)
            print(y.shape)  # Output: torch.Size([1, 256, 20, 20])
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
        Initializes the Expand module to increase spatial dimensions by redistributing channels, with an optional gain factor.
        
        Args:
            gain (int): Factor by which spatial dimensions (width and height) are expanded while reducing channel depth
                accordingly. Defaults to 2.
        
        Returns:
            None
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """
        Processes input tensor `x` to expand its spatial dimensions by redistributing channels, ensuring correctness of the 
        channel and gain relationship.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor with expanded spatial dimensions and adjusted channels.
        
        Notes:
            This function is critical for architectures where spatial resolution needs to be increased while appropriately 
            managing channel information. The effectiveness relies on the fact that the channel dimension is divisible by 
            `gain` squared. Ensure input dimensions are compatible before utilization.
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
            dimension (int): The dimension along which to concatenate the tensors. Default is 1.
        
        Returns:
            None. This is an initializer and does not return a value.
        
        Examples:
            ```python
            import torch
            from ultralytics import Concat
        
            # Initialize Concat layer
            concat_layer = Concat(dimension=1)
        
            # Create sample tensors
            tensor1 = torch.randn(2, 3)
            tensor2 = torch.randn(2, 3)
        
            # Use the Concat layer
            result = concat_layer([tensor1, tensor2])
            ```
        """
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """
        Concatenates a list of tensors along a specified dimension.
        
        Args:
            x (list[torch.Tensor]): List of tensors to concatenate. All tensors must have the same shape, 
                                    except for the dimension along which they will be concatenated.
        
        Returns:
            torch.Tensor: A single tensor obtained by concatenating the input tensors along the specified dimension.
        
        Examples:
            ```python
            concat_layer = Concat(dimension=1)
            tensor1 = torch.randn(3, 4, 5)
            tensor2 = torch.randn(3, 4, 5)
            output = concat_layer([tensor1, tensor2])
            print(output.shape)  # Output shape: torch.Size([3, 8, 5])
            ```
        """
        return torch.cat(x, self.d)


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights="yolov5s.pt", device=torch.device("cpu"), dnn=False, data=None, fp16=False, fuse=True):
        """
        Initializes the DetectMultiBackend class for YOLOv5 to support inference across different backends including 
        PyTorch, ONNX Runtime, TensorRT, TensorFlow, and others.
        
        Args:
            weights (str | Path | list): Model weights path or URL. It supports extensions such as .pt, .onnx, .engine, etc. 
                                         Defaults to "yolov5s.pt".
            device (torch.device): Torch device to execute the model, typically 'cpu' or 'cuda'. Defaults to torch.device("cpu").
            dnn (bool): Flag to indicate whether to use OpenCV DNN as ONNX backend. Defaults to False.
            data (str | dict | None): Path to data.yaml file or a dictionary with class names. Defaults to None.
            fp16 (bool): Flag to indicate inference with FP16 precision for supported backends. Defaults to False.
            fuse (bool): Flag that determines whether to fuse Conv2d + BatchNorm2d layers for optimized inference. Defaults to True.
        
        Returns:
            None: This function initializes the class instance.
        
        Note:
            This function supports loading and managing different model formats for YOLOv5, such as PyTorch, TorchScript, 
            ONNX, OpenVINO, CoreML, TensorRT, TensorFlow, PaddlePaddle, and more. Ensure that the necessary libraries for the 
            selected backend are installed.
        
        Example:
            ```python
            from ultralytics import DetectMultiBackend
            model = DetectMultiBackend(weights='path/to/model.onnx', device=torch.device('cuda'), dnn=False, fp16=True)
            ```
        
        Links:
            - [Ultralytics YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
            - [Model Export Formats](https://github.com/ultralytics/yolov5/issues/1173)
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
        Performs YOLOv5 inference on input images using multiple backend options, with augmentation and visualization.
        
        Args:
            im (torch.Tensor): Input image tensor with shape (B, C, H, W).
            augment (bool, optional): If True, applies test-time augmentation. Defaults to False.
            visualize (bool, optional): If True, enhances visualization of the detection steps. Defaults to False.
        
        Returns:
            torch.Tensor | list[torch.Tensor] | np.ndarray | list[np.ndarray]: Prediction results, which can be a single tensor, 
            a list of tensors, or a numpy array, depending on the backend and output formats used.
        
        Notes:
            - For customized metadata loading and backend-specific configurations, refer to the initialization parameters.
            - Backend support includes PyTorch, TorchScript, ONNX, OpenVINO, CoreML, TensorRT, TensorFlow (various formats),
              PaddlePaddle, and NVIDIA Triton Inference Server.
            - Autodetects batch sizes and dynamic shapes for platforms like TensorRT to accommodate input variability.
            - Converts input to appropriate dtype (FP16/FP32) and layout (BCHW to BHWC) based on the backend compatibility.
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
        Converts a NumPy array to a PyTorch tensor, ensuring compatibility with the current device.
        
        Args:
            x (np.ndarray): The input NumPy array to be converted.
        
        Returns:
            torch.Tensor: The resulting PyTorch tensor on the same device as the model.
        
        Note:
            This method facilitates seamless integration of NumPy-based operations within the PyTorch framework, crucial for scenarios involving multi-backend operations across various frameworks (e.g., TensorFlow, ONNX, OpenVINO, CoreML). Please ensure that the input array `x` is properly formatted according to the expected tensor shape of subsequent operations.
        
        Example:
            ```python
            import numpy as np
            from ultralytics import DetectMultiBackend
        
            detector = DetectMultiBackend(weights='yolov5s.pt')
            numpy_array = np.random.randn(1, 3, 640, 640)  # Example NumPy array
            tensor = detector.from_numpy(numpy_array)
            print(tensor.shape)  # torch.Size([1, 3, 640, 640])
            ```
        """
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        Perform a single inference warmup to initialize model weights with a specified image size.
        
        Args:
          imgsz (tuple[int, int, int, int]): Size of the warmup image tensor to perform inference on, typically in format 
             (batch_size, channels, height, width). Default is (1, 3, 640, 640).
        
        Returns:
          None
        """
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p="path/to/model.pt"):
        """
        Determines the model type based on the file extension or URL, supporting a variety of model formats.
        
        Args:
            p (str): The file path or URL to the model. 
        
        Returns:
            tuple: A tuple of boolean values indicating the type of model. The values correspond to the 
                following formats: (pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton).
        
        Example:
            ```python
            model_type = DetectMultiBackend._model_type("path/to/model.onnx")
            print(model_type)  # Output: (False, False, True, False, False, False, False, False, False, False, False, False, False)
            ```
        
        Notes:
            See more on export formats here: https://github.com/ultralytics/ultralytics
        
            It uses utility functions from `utils.downloads` and `export` modules to assist in the determination.
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
        Loads metadata from a specified YAML file.
        
        Args:
          f (Path): Path to the metadata YAML file.
        
        Returns:
          tuple[int, dict[int | str, str]]: Returns a tuple containing the stride (int) and the names dictionary where keys 
                                            are class indices (int | str) and values are class names (str). Returns None if 
                                            the file does not exist.
        
        Notes:
          The YAML file is expected to contain a "stride" field (int) and a "names" field (dict) where the keys are class 
          indices and values are class names. Example structure:
          ```
          stride: 32
          names: {0: 'class0', 1: 'class1'}
          ```
        
        Example:
          ```python
          metadata = DetectMultiBackend._load_metadata(Path("path/to/meta.yaml"))
          if metadata:
              stride, names = metadata
              print(stride)  # Outputs the stride value
              print(names)  # Outputs the class names dictionary
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
        Initializes the YOLOv5 model for inference, setting up necessary attributes and preparing the model for evaluation.
               
        Args:
            model (nn.Module): The model to be wrapped for auto shape handling. Typically, a YOLOv5 model instance.
            verbose (bool): A flag to control the verbosity of the initialization. Defaults to True to display initialization info.
        
        Returns:
            None
        
        Notes:
            This class is designed to make the YOLOv5 model input robust, supporting various image formats (cv2, numpy, PIL, torch).
            It includes preprocessing, inference, and Non-Maximum Suppression (NMS) functionalities. The class also handles
            automatic mixed precision (AMP) inference and allows optional filtering by specified classes.
        
        Examples:
            ```python
            model = torch.load('yolov5s.pt')['model'].float().fuse().eval()  # Load a pretrained YOLOv5 model
            autos = AutoShape(model)  # Add AutoShape to handle different input formats
            ```
        
        References:
            - https://github.com/ultralytics/ultralytics
        
        Attributes:
            conf (float): NMS confidence threshold.
            iou (float): NMS IoU threshold.
            agnostic (bool): NMS class-agnostic flag.
            multi_label (bool): NMS multiple labels per box flag.
            classes (list | None): Optional list to filter detections by class.
            max_det (int): Maximum number of detections per image.
            amp (bool): Automatic Mixed Precision (AMP) inference flag.
            model (nn.Module): The underlying YOLOv5 model, set to evaluation mode.
            dmb (bool): Flag indicating if the model is an instance of DetectMultiBackend.
            pt (bool): Flag indicating if the model is a PyTorch model.
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
        Applies a function to the module, including model parameters and buffers but excluding irrelevant tensors.
        
        Args:
            fn (Callable): Function to apply to each model parameter and buffer.
        
        Returns:
            self (nn.Module): The model with the function applied to its appropriate parameters and buffers.
        
        Example:
            ```python
            # Applying the function to move the model to GPU
            model._apply(lambda t: t.cuda())
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
        Performs inference on various input types including file paths, URIs, OpenCV images, PIL images, numpy arrays, and 
        torch tensors, with optional augmentation and profiling.
        
        Args:
            ims (str | Path | np.ndarray | Image.Image | torch.Tensor | list): Inputs for inference, which can be a single image 
              file path, URI, OpenCV image, PIL image, numpy array, torch tensor, or a list of such entries.
            size (int | tuple): The target size for resizing input images. If an integer is provided, it sets both width and 
              height. If a tuple, the first value is the height and the second is the width.
            augment (bool): A flag to indicate if augmentation should be applied during inference. Default is False.
            profile (bool): A flag to turn on profiling for performance measurement. Default is False.
        
        Returns:
            list: A list of detections per input image, with each detection containing bounding boxes, confidence scores, and 
              class labels. The format can vary based on input type and optional configurations.
        
        Notes:
            This method supports a wide range of input types:
            - For file paths or URIs, `ims` should be a string or pathlib.Path object.
            - For images read by OpenCV, `ims` should be a NumPy array in HWC BGR format.
            - For PIL images, `ims` should be an Image.Image object.
            - For raw image data, `ims` should be a NumPy array in HWC format, with RGB ordering.
            - For batched data, `ims` should be a torch tensor in BCHW format.
        
        Example:
            ```python
            from ultralytics import YOLOv5
            
            model = YOLOv5('path/to/model.pt')
            results = model.autoshape.forward(['image1.jpg', 'image2.png'], size=640)
            ```
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
        Initializes the Detections object for YOLOv5 inference results.
        
        Args:
            ims (list[np.ndarray]): List of input images in numpy array format.
            pred (list[torch.tensor]): List of tensors containing predictions for each image.
            files (list[str]): List of filenames corresponding to the images.
            times (tuple[float, float, float], optional): Profiling times for various stages (pre-processing, inference, 
                                                           and post-processing), defaults to (0, 0, 0).
            names (list[str], optional): List of class names for object detection classes. Defaults to None.
            shape (tuple[int, int, int, int], optional): Shape of input images as (batch, channel, height, width). Defaults to None.
        
        Notes:
            - The normalizations `gn` are precomputed values to facilitate conversion between various coordinate formats.
            - Timestamp calculations are performed to provide insights into the time taken per image in milliseconds.
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
        Executes model predictions, displaying and/or saving outputs with optional crops and labels.
        
        Args:
            pprint (bool): If True, pretty-prints the detection results.
            show (bool): If True, displays the images with annotated predictions.
            save (bool): If True, saves the images with annotated predictions.
            crop (bool): If True, crops the detected objects and saves them.
            render (bool): If True, render the detected objects for display.
            labels (bool): If True, include labels in the annotated images.
            save_dir (Path): Directory for saving annotated images and crops.
        
        Returns:
            str: If `pprint` is True, returns a formatted string describing the detections and timing.
        
        Example:
            ```python
            detections = Detections(ims, pred, files)
            detections._run(pprint=True, show=True, save=True, crop=True, render=True, labels=True, save_dir=Path("./"))
            ```
        
            Detection results are printed, displayed, saved with bounding boxes, and cropped. 
            Rendered images are also shown if `show` is True.
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
            labels (bool): Whether to display labels on the detections. Defaults to True.
        
        Returns:
            None: This method does not return any value. It simply displays the images with detection results.
        
        Examples:
            Show detection results with labels:
            ```python
            detections.show(labels=True)
            ```
        
            Show detection results without labels:
            ```python
            detections.show(labels=False)
            ```
        
        Notes:
            This function is environment-aware and includes error-handling to notify if the environment does not support image display.
        
            In a Jupyter Notebook environment, it will use `IPython.display.display` to show the images.
        """
        self._run(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir="runs/detect/exp", exist_ok=False):
        """
        Saves detection results with optional labels to a specified directory.
        
        Args:
          labels (bool): Whether to include labels in saved images. Defaults to True.
          save_dir (str | Path): Directory where results will be saved. Defaults to 'runs/detect/exp'.
          exist_ok (bool): If True, existing save_dir will be used without incrementing. Defaults to False.
        
        Returns:
          None
        
        Example:
          Save detection results to a custom directory:
            
          ```python
          detections.save(labels=True, save_dir="my_detection_output", exist_ok=True)
          ```
        """
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
        self._run(save=True, labels=labels, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir="runs/detect/exp", exist_ok=False):
        """
        Uses detection results to crop regions of interest.
        
        Args:
            save (bool): Whether to save the cropped regions. Default is True.
            save_dir (str | Path): Directory to save cropped images if `save` is True. Default is 'runs/detect/exp'.
            exist_ok (bool): Whether the save directory can exist without incrementing. Default is False.
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries, each containing crop metadata such as 'box', 'conf', 'cls', 'label', and 'im'.
        
        Example:
            ```python
            detections = Detections(ims, pred, files)
            crops = detections.crop(save=True, save_dir='runs/detect/crops')
            ```
            """
            save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
            return self._run(crop=True, save=save, save_dir=save_dir)
        """
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
        return self._run(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self, labels=True):
        """
        Renders detection results on images.
        
        Args:
            labels (bool): Flag indicating whether to render labels on the detected bounding boxes. Default is True.
        
        Returns:
            None: This method modifies the images in-place with rendered detections.
        
        Example:
            ```python
            detections = model(image)  # Perform detection on image
            detections.render()  # Render results on the image
            ```
        """
        self._run(render=True, labels=labels)  # render results
        return self.ims

    def pandas(self):
        """
        Returns detections as pandas DataFrames for various box formats (xyxy, xyxyn, xywh, xywhn).
        
        Returns:
            Detections: An object containing four lists with pandas DataFrames:
                - xyxy: DataFrame with columns ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'].
                - xyxyn: DataFrame with columns ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'], normalized.
                - xywh: DataFrame with columns ['xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'].
                - xywhn: DataFrame with columns ['xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'],
                        normalized.
        
        Example:
            ```python
            results = model(imgs)
            print(results.pandas().xyxy[0])
            ```
        
        Note:
            Use the DataFrame object to manage and manipulate detection results more efficiently, supporting numerous pandas
            operations for advanced analysis and processing.
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
        Converts a `Detections` object to a list of single-image detection results.
        
        Example:
            Use this method to facilitate iteration over individual image results:
            ```python
            for result in results.tolist():
                print(result.xyxy)
            ```
        
        Returns:
            list[Detections]: A list of `Detections` objects, each corresponding to a single image result.
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
        Prints a summary of detection results.
        
        This method logs a textual summary of the detection results, including the number of detected objects per image and their respective classes.
        
        Attributes:
            None
        
        Returns:
            None
        
        Examples:
            ```python
            from ultralytics import Detections
        
            detections = Detections(...)
            detections.print()
            ```
        
        Notes:
            This function calls the internal `_run` method with `pprint=True`, which generates the summary text and logs it via the `LOGGER`.
            
            For more information, visit https://github.com/ultralytics/ultralytics.
        """
        LOGGER.info(self.__str__())

    def __len__(self):
        """
        Returns the number of detections.
        
            Returns:
                int: The number of detection results stored in the Detections object.
        
            Example:
                ```python
                detections = model(image)
                num_detections = len(detections)
                print(f"Number of detection results: {num_detections}")
                ```
        """
        return self.n

    def __str__(self):
        """
        __str__()
        
        Represents the Detections object as a string, giving a human-readable summary of detection results.
        
        Returns:
            (str): A string summarizing the detection results, including image dimensions, object counts, and processing times.
        
        Examples:
            ```python
            detections = model(img)
            print(detections)
            # Expected Output:
            # "image 1/1: 640x640 3 persons, 1 car
            # Speed: 10.4ms pre-process, 23.2ms inference, 2.1ms NMS per image at shape (1, 3, 640, 640)"
            ```
        """
        return self._run(pprint=True)  # print results

    def __repr__(self):
        """
        Returns a string representation of the Detections object including class results.
        
        Returns:
            str: A formatted string representation of the detection results.
        
        Examples:
            ```python
            detections = Detections(ims, pred, files, times, names, shape)
            print(repr(detections))
            ```
        
            This would output:
            ```
            image 1/1: 640x480 1 person, 1 dog
            Speed: 2.0ms pre-process, 10.0ms inference, 1.0ms NMS per image at shape (1, 3, 640, 640)
            ```
        """
        return f"YOLOv5 {self.__class__} instance\n" + self.__str__()


class Proto(nn.Module):
    # YOLOv5 mask Proto module for segmentation models
    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv5 mask Proto module for segmentation models.
        
        Args:
            c1 (int): Number of input channels.
            c_ (int): Number of intermediate channels in the proto layer (default is 256).
            c2 (int): Number of output channels in the mask layer (default is 32).
        
        Returns:
            None: This function does not return a value; it initializes layers of the Proto module.
        
        Example:
            ```python
            proto = Proto(c1=512, c_=256, c2=32)
            ```
        
        Note:
            The Proto module is part of the YOLOv5 architecture, specifically utilized for segmentation tasks. It consists of a 
            convolutional layer followed by an upsampling layer and another convolutional layer for feature extraction and
            refinement. This module is integral to YOLOv5's ability to perform instance segmentation.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """
        Performs a forward pass using convolutional layers and upsampling on the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Transformed tensor with upsampled spatial dimensions and adjusted channel depth.
        """
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Classify(nn.Module):
    # YOLOv5 classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, dropout_p=0.0
    ):  # ch_in, ch_out, kernel, stride, padding, groups, dropout probability
        """
        Initializes a YOLOv5 classifier head with convolution, pooling, and optional dropout layers for classification tasks.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels (number of classes).
            k (int, optional): Kernel size for the convolutional layer. Defaults to 1.
            s (int, optional): Stride size for the convolutional layer. Defaults to 1.
            p (int | None, optional): Padding size for the convolutional layer. Defaults to None.
            g (int, optional): Number of groups for the convolutional layer. Defaults to 1.
            dropout_p (float, optional): Probability of an element to be zeroed in dropout layer. Defaults to 0.0.
        
        Returns:
            None: This method initializes the layers of the classifier.
        
        Notes:
            Uses default kernel and stride values for defining the convolutional layer.
            Padding is automatically determined if not provided.
            
        Examples:
            ```python
            from yolov5.models.common import Classify
        
            # Initialize classifier with input channels, output channels, and dropout probability
            classifier = Classify(c1=512, c2=10, dropout_p=0.5)
            ```
        
            Model is defined with efficientnet_b0 size for convolution and uses average pooling and dropout.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=dropout_p, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """
        Performs a forward pass through the YOLOv5 classification head, transforming input tensors into predicted class logits.
        
        Args:
            x (torch.Tensor | list[torch.Tensor]): Input tensor or list of concatenated input tensors with shape (batch_size,
            in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor containing predicted class logits of shape (batch_size, num_classes).
        
        Example:
        ```python
        model = Classify(c1=640, c2=10)
        input_tensor = torch.randn(1, 640, 20, 20)
        output = model.forward(input_tensor)
        print(output.shape)  # torch.Size([1, 10])
        ```
        """
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
