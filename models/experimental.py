# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Experimental modules."""

import math

import numpy as np
import torch
import torch.nn as nn

from utils.downloads import attempt_download


class Sum(nn.Module):
    """Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070."""

    def __init__(self, n, weight=False):
        """
        Initialize the Sum module which performs a weighted sum of outputs from multiple layers.
        
        Args:
            n (int): Number of input layers.
            weight (bool): If True, apply learnable weights to each input layer.
        
        Returns:
            (None)
        
        Example:
            ```python
            sum_module = Sum(n=3, weight=True)
            ```
        
        Note:
            This module is useful for merging features from different layers, as described in 
            https://arxiv.org/abs/1911.09070.
        """
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        """
        Aggregates input tensors using a weighted or unweighted sum, based on initialization parameters.
        
        Args:
            x (list[torch.Tensor]): A list of input tensors to be summed, with each tensor having shape (N, ...).
        
        Returns:
            (torch.Tensor): The resulting tensor after summing, with the same shape (N, ...) as the input tensors.
        
        Example:
            ```python
            sum_module = Sum(n=3, weight=True)
            input_tensors = [torch.randn(5, 10), torch.randn(5, 10), torch.randn(5, 10)]
            output_tensor = sum_module(input_tensors)
            ```
        """
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    """Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595."""

    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        """
        Initialize MixConv2d with mixed depth-wise convolutional layers for enhanced computational efficiency and accuracy.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (tuple[int]): Tuple of kernel sizes to be used in mixed convolutions.
            s (int): Stride for convolutions.
            equal_ch (bool): Whether to use equal channels per group (True) or equal weight per group (False).
        
        Returns:
            (torch.Tensor): Output tensor after applying mixed depth-wise convolution and batch normalization.
        
        Example:
            ```python
            mixconv_layer = MixConv2d(64, 128, k=(1, 3, 5), s=1, equal_ch=True)
            input_tensor = torch.randn(1, 64, 224, 224)
            output_tensor = mixconv_layer(input_tensor)
            ```
        """
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1e-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)]
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Perform a forward pass with MixConv2d layer to apply mixed depth-wise convolutions.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W), where N is the batch size, C is the number of channels, 
                H is the height, and W is the width.
        
        Returns:
            (torch.Tensor): Resulting tensor after applying mixed depth-wise convolutions, batch normalization, 
                and SiLU activation.
        
        Example:
            ```python
            mixconv = MixConv2d(c1=16, c2=32, k=(1, 3, 5), s=1, equal_ch=True)
            x = torch.rand(1, 16, 64, 64)
            y = mixconv(x)
            print(y.shape)  # Expected output shape: (1, 32, 64, 64)
            ```
        
        Note:
            This function is based on the mixed depth-wise convolution from the paper: 
            https://arxiv.org/abs/1907.09595.
        """
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """
        Initializes an ensemble of models, typically used for combining predictions from multiple YOLO models to improve 
        inference accuracy.
        
        Args:
            None
        
        Returns:
            (None)
        
        Example:
            ```python
            model_ensemble = Ensemble()
            ```
        """
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """
        Perform a forward pass aggregating outputs from an ensemble of models.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) where N is the batch size, C is the number of channels, 
                H is the height, and W is the width.
            augment (bool): If True, perform augmented inference.
            profile (bool): If True, profile the forward pass for performance.
            visualize (bool): If True, visualize features of the model.
        
        Returns:
            (torch.Tensor): Aggregated output tensor from the ensemble of models, with shape (N, C', H, W) where C' is 
                the concatenated channels from individual model outputs.
        
        Example:
            ```python
            ensemble_model = Ensemble()
            input_tensor = torch.randn(1, 3, 224, 224)
            output_tensor = ensemble_model(input_tensor)
            print(output_tensor.shape)  # Expected output shape will depend on individual models in the ensemble
            ```
        
        Note:
            This function assumes that each model in the ensemble produces outputs with shapes that can be concatenated 
            along the channel dimension.
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, device=None, inplace=True, fuse=True):
    """
    Load and prepare a YOLOv5 model or ensemble of models from specified weights, supporting device placement and model fusion.
    
    Args:
        weights (list[str] | str): Path(s) to model weights. Can be a list of multiple weight paths or a single path.
        device (torch.device | None): Device for model loading. If None, defaults to 'cpu'.
        inplace (bool): Whether to use inplace operations in the model (e.g., for activation layers).
        fuse (bool): Whether to fuse model layers to achieve better inference performance.
    
    Returns:
        (nn.Module): A PyTorch model, either a single YOLOv5 model or an ensemble of models for inference.
    
    Example:
        ```python
        model = attempt_load(weights='yolov5s.pt', device=torch.device('cuda'), inplace=True, fuse=True)
        ```
    
    Notes:
        This function ensures compatibility with older model formats by performing necessary updates (e.g., stride and
        anchor grid adjustments).
    """
    from models.yolo import Detect, Model

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location="cpu")  # load
        ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, "stride"):
            ckpt.stride = torch.tensor([32.0])
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval())  # model in eval mode

    # Module updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f"Models have different class counts: {[m.nc for m in model]}"
    return model
