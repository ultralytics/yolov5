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
        Initializes the Sum module to compute a weighted or unweighted sum of multiple input layers.

        Args:
            n (int): The number of input layers to be summed. Must be 2 or more.
            weight (bool): Flag indicating whether to apply weights to the summed layers. Defaults to False.

        Returns:
            None

        Note:
            For more information, refer to the paper at https://arxiv.org/abs/1911.09070

        Example:
            ```python
            sum_module = Sum(n=3, weight=True)
            ```
        """
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        """
        Processes input through a customizable weighted sum of `n` inputs, optionally applying learned weights.

        Args:
          x (list[torch.Tensor]): List of input tensors with shapes compatible for summation.

        Returns:
          torch.Tensor: The resultant tensor after either plain or weighted summation.

        Note:
          The method applies weights only if the `weight` attribute was enabled during initialization. Weights are learned
          parameters with values constrained between 0 and 2 through a sigmoid function.

        Example:
          ```python
          import torch
          from your_module import Sum

          sum_module = Sum(n=3, weight=True)  # Initialize Sum module with weight=True
          inputs = [torch.randn(1, 3, 256, 256) for _ in range(3)]  # Example inputs
          output = sum_module(inputs)  # Process through Sum.forward()
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
        Initializes MixConv2d with mixed depth-wise convolutional layers, specifying input/output channels, kernel
        sizes, stride, and channel distribution strategy.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (List[int] | Tuple[int] | int): Kernel sizes for each group, default is (1, 3).
            s (int): Stride value, default is 1.
            equal_ch (bool): Flag to decide channel distribution strategy; True for equal channels per group, False for equal number of weights per group, default is True.

        Returns:
            None

        Notes:
            Refer to the Mixed Depth-wise Convolutional Networks paper for more details: https://arxiv.org/abs/1907.09595

        Example:
            ```python
            mix_conv2d = MixConv2d(c1=32, c2=64, k=(1, 3, 5), s=2, equal_ch=False)
            output = mix_conv2d(input_tensor)
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
        Performs a forward pass through the mixed depth-wise convolutional layers, applies batch normalization, and SiLU
        activation.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H, W), where N is the batch size, C_in is the number
                of input channels, and H, W are the height and width of the input feature maps.

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H_out, W_out), where C_out is the number of output channels,
                and H_out, W_out are the height and width of the output feature maps after convolution, batch normalization,
                and SiLU activation.

        Examples:
            ```python
            # Initialize the MixConv2d layer
            mixconv = MixConv2d(c1=64, c2=128, k=(3, 5), s=1, equal_ch=True)

            # Create a random input tensor with shape (N, C_in, H, W)
            input_tensor = torch.randn(1, 64, 224, 224)

            # Perform the forward pass
            output_tensor = mixconv.forward(input_tensor)

            # output_tensor will have shape (1, 128, 224, 224)
            ```
        """
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """
        Initializes an ensemble of models to be used for aggregated predictions.

        This class manages a collection of models. It can be used to combine the outputs of multiple models to improve prediction accuracy through techniques like averaging or voting.

        Args:
            None

        Returns:
            None

        Notes:
            The Ensemble class inherits from `torch.nn.ModuleList`, so it can be used as a standard PyTorch module list.
            It allows easy integration and manipulation of different models within the PyTorch framework.

        Example:
            ```python
            import torch.nn as nn

            # Instantiate the ensemble
            models = Ensemble()

            # Add models to the ensemble
            models.append(nn.Linear(10, 5))
            models.append(nn.Conv2d(3, 16, 3, 1, 1))

            # Use the ensemble
            for model in models:
                print(model)
            ```
        """
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """
        Performs forward pass aggregating outputs from an ensemble of models.

        Args:
            x (torch.Tensor): The input tensor to be processed through the ensemble of models.
            augment (bool): If True, enables test time augmentation. Default is False.
            profile (bool): If True, tracks and reports the computation time and memory usage. Default is False.
            visualize (bool): If True, enables visualization of intermediate model outputs. Default is False.

        Returns:
            torch.Tensor: Aggregated output tensor after processing through all models in the ensemble. Output shape will depend
            on the aggregation method used (concatenation, mean, or max).

        Examples:
            ```python
            # Assuming `ensemble` is an instance of Ensemble with loaded models
            input_tensor = torch.randn(1, 3, 640, 640)  # Example input tensor
            output = ensemble(input_tensor, augment=True)  # Perform forward pass with augmentation
            ```
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, device=None, inplace=True, fuse=True):
    """
    Loads and fuses an ensemble or single YOLOv5 model from provided weights, handling device placement and model
    adjustments.

    Args:
        weights (str | list[str]): Path to model weights file(s). Can be a single path or list of paths.
        device (torch.device | None): The device on which to load the model. If None, defaults to 'cpu' (optional).
        inplace (bool): Whether to set layers to inplace mode for reduced memory usage (optional, default is True).
        fuse (bool): Whether to fuse model Conv2d + BatchNorm2d layers for inference (optional, default is True).

    Returns:
        nn.Module: Loaded model, either a single YOLOv5 model or an ensemble of models, ready for inference or further
        training.

    Example:
        ```python
        from ultralytics import attempt_load

        model = attempt_load(['yolov5s.pt', 'yolov5m.pt'], device='cuda')
        ```

    Notes:
        - Ensure model weights are accessible from the provided path(s).
        - Device should be specified for optimal model performance, especially during inference.
        - The method handles compatibility updates for various PyTorch and YOLOv5 versions.

    For more information, visit the official Ultralytics YOLOv5 repository:
    https://github.com/ultralytics/yolov5
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
