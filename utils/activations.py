# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Activation functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    """Applies the Sigmoid-weighted Linear Unit (SiLU) activation function, also known as Swish."""

    @staticmethod
    def forward(x):
        """
        Applies the Sigmoid-weighted Linear Unit (SiLU) activation function.

        https://arxiv.org/pdf/1606.08415.pdf.
        """
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):
    """Applies the Hardswish activation function, which is efficient for mobile and embedded devices."""

    @staticmethod
    def forward(x):
        """
        Applies the Hardswish activation function, compatible with TorchScript, CoreML, and ONNX.

        Equivalent to x * F.hardsigmoid(x)
        """
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0  # for TorchScript, CoreML and ONNX


class Mish(nn.Module):
    """Mish activation https://github.com/digantamisra98/Mish."""

    @staticmethod
    def forward(x):
        """Applies the Mish activation function, a smooth alternative to ReLU."""
        return x * F.softplus(x).tanh()


class MemoryEfficientMish(nn.Module):
    """Efficiently applies the Mish activation function using custom autograd for reduced memory usage."""

    class F(torch.autograd.Function):
        """Implements a custom autograd function for memory-efficient Mish activation."""

        @staticmethod
        def forward(ctx, x):
            """Applies the Mish activation function, a smooth ReLU alternative, to the input tensor `x`."""
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            """Computes the gradient of the Mish activation function with respect to input `x`."""
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        """Applies the Mish activation function to the input tensor `x`."""
        return self.F.apply(x)


class FReLU(nn.Module):
    """FReLU activation https://arxiv.org/abs/2007.11824."""

    def __init__(self, c1, k=3):  # ch_in, kernel
        """Initializes FReLU activation with channel `c1` and kernel size `k`."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        """
        Applies FReLU activation with max operation between input and BN-convolved input.

        https://arxiv.org/abs/2007.11824
        """
        return torch.max(x, self.bn(self.conv(x)))


class AconC(nn.Module):
    """
    ACON activation (activate or not) function.

    AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    See "Activate or Not: Learning Customized Activation" https://arxiv.org/pdf/2009.04759.pdf.
    """

    def __init__(self, c1):
        """Initializes AconC with learnable parameters p1, p2, and beta for channel-wise activation control."""
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, c1, 1, 1))

    def forward(self, x):
        """Applies AconC activation function with learnable parameters for channel-wise control on input tensor x."""
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(self.beta * dpx) + self.p2 * x


class MetaAconC(nn.Module):
    """
    ACON activation (activate or not) function.

    AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    See "Activate or Not: Learning Customized Activation" https://arxiv.org/pdf/2009.04759.pdf.
    """

    def __init__(self, c1, k=1, s=1, r=16):
        """Initializes MetaAconC with params: channel_in (c1), kernel size (k=1), stride (s=1), reduction (r=16)."""
        super().__init__()
        c2 = max(r, c1 // r)
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.fc1 = nn.Conv2d(c1, c2, k, s, bias=True)
        self.fc2 = nn.Conv2d(c2, c1, k, s, bias=True)
        # self.bn1 = nn.BatchNorm2d(c2)
        # self.bn2 = nn.BatchNorm2d(c1)

    def forward(self, x):
        """Applies a forward pass transforming input `x` using learnable parameters and sigmoid activation."""
        y = x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True)
        # batch-size 1 bug/instabilities https://github.com/ultralytics/yolov5/issues/2891
        # beta = torch.sigmoid(self.bn2(self.fc2(self.bn1(self.fc1(y)))))  # bug/unstable
        beta = torch.sigmoid(self.fc2(self.fc1(y)))  # bug patch BN layers removed
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(beta * dpx) + self.p2 * x
