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
        Initialize the Sum module to aggregate outputs from multiple layers, optionally with weights.

        Args:
            n (int): Number of layers to sum. Must be 2 or more.
            weight (bool): If True, applies weights to the inputs before summing.

        Returns:
            None

        Notes:
            Refer to "Weighted sum of 2 or more layers" at https://arxiv.org/abs/1911.09070 for detailed insights
            and usage scenarios.
        """
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        """
        Compute a weighted or unweighted sum of input tensors.

        Args:
            x (list[torch.Tensor]): List of input tensors to be summed, with each tensor having the same shape (N, D).

        Returns:
            (torch.Tensor): The resulting tensor after summing the input tensors, maintaining the same shape (N, D).

        Example:
            ```python
            sum_layer = Sum(n=3, weight=False)
            inputs = [torch.rand(1, 10), torch.rand(1, 10), torch.rand(1, 10)]
            result = sum_layer.forward(inputs)
            ```

        Note:
            If `weight` is set to True when initializing the class, weights will be applied to the inputs before summing.
            For more information, refer to "Weighted sum of 2 or more layers" at https://arxiv.org/abs/1911.09070.
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
        Initialize the MixConv2d module, handling mixed depth-wise convolutional operations.

        Args:
            c1 (int): Number of input channels (C1).
            c2 (int): Number of output channels (C2).
            k (tuple[int]): Kernel sizes for the convolutional layers.
            s (int): Stride value for the convolutional layers.
            equal_ch (bool): Flag to determine if channels are distributed equally. True for equal channels per group, False
                for equal weight.numel() per group.

        Example:
            ```python
            mixconv = MixConv2d(c1=32, c2=64, k=(1, 3, 5), s=1, equal_ch=True)
            output = mixconv(input_tensor)
            ```

        Note:
            The `MixConv2d` layer applies multiple depth-wise convolutions with different kernel sizes in parallel, which
            can capture multi-scale features within a single layer. This technique is particularly useful for improving
            spatial feature extraction and reducing model complexity.

            Further reading: https://arxiv.org/abs/1907.09595
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
        Perform forward pass by applying mixed depth-wise convolutions followed by batch normalization and SiLU
        activation.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W) where N is the batch size, C is the number of channels,
                H is the height, and W is the width.

        Returns:
            (torch.Tensor): Output tensor after applying mixed convolutions, batch normalization, and SiLU activation,
                maintaining the shape (N, C', H', W') where C' is the output channels based on the convolutional layer
                configuration.

        Example:
            ```python
            mixconv = MixConv2d(c1=32, c2=64, k=(1, 3), s=1)
            x = torch.randn(16, 32, 128, 128)
            output = mixconv(x)
            ```
        """
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """
        Initializes an ensemble of models for combined inference and aggregated predictions.

        Example:
            ```python
            ensemble = Ensemble()
            model1 = MyModel1()
            model2 = MyModel2()
            ensemble.append(model1)
            ensemble.append(model2)
            ```
        """
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """
        Aggregates outputs from multiple models in the ensemble by concatenating them during the forward pass.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W) where N is the batch size, C is the number of channels,
                H is the height, and W is the width.
            augment (bool): Flag to apply test-time augmentation (TTA) during inference. Default is False.
            profile (bool): If True, enables profiling of the forward pass. Default is False.
            visualize (bool): If True, enables visualization of model predictions. Default is False.

        Returns:
            (torch.Tensor): Aggregated output tensor from the ensemble models, with shape dependent on the number of models
                and their architectures.

        Example:
            ```python
            from ultralytics import Ensemble
            import torch

            # Initialize the ensemble
            ensemble = Ensemble()
            # Assume models are already added to the ensemble

            # Create a dummy input tensor
            x = torch.randn(8, 3, 640, 640)  # Example input for 8 images of 3 channels and 640x640 resolution

            # Perform forward pass
            output = ensemble.forward(x, augment=False, profile=False, visualize=False)
            ```
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, device=None, inplace=True, fuse=True):
    """
    Loads and fuses a YOLOv5 model or an ensemble of models from provided weights, adjusting device placement and model
    attributes for optimal performance.

    Args:
        weights (str | list[str]): Path(s) to model weight file(s). It can be a single path or a list of paths.
        device (torch.device | None, optional): Device to load the model on. If None, loads on CPU by default.
        inplace (bool, optional): If True, enables inplace operations in certain layers like activation layers.
            Defaults to True.
        fuse (bool, optional): Whether to fuse Conv2d + BatchNorm2d layers for speedup during inference. Defaults to True.

    Returns:
        (torch.nn.Module): Loaded YOLOv5 model or an ensemble of models loaded onto the specified device.

    Example:
        ```python
        # Load a single model weight
        model = attempt_load('yolov5s.pt')

        # Load an ensemble of models
        model = attempt_load(['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt'])
        ```

    Note:
        - This function ensures compatibility and performance optimization by adjusting attributes and configurations of the
          loaded model(s).
        - If `fuse` is set to True, it will fuse Conv2d and BatchNorm2d layers within the model(s) to speed up inference.
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
