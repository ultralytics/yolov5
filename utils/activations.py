"""Activation functions."""

import torch
from torch import nn as nn
from torch.nn import functional as F


class SiLU(nn.Module):
    """Export-friendly version of nn.SiLU().

    SiLU https://arxiv.org/pdf/1905.02244.pdf

    Args:
        nn (module): torch.nn
    """

    @staticmethod
    def forward(x):
        """Forward Propagation.

        Args:
            x (torch.Tensor): Tensor activated element-wise

        Returns:
            [type]: [description]
        """
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):
    """Export-friendly version of nn.Hardswish().

    Args:
        nn (module): torch.nn
    """

    @staticmethod
    def forward(x):
        """Forward Propagation.

        Args:
            x (torch.Tensor): Tensor activated element-wise

        Returns:
            [type]: [description]
        """
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX


class MemoryEfficientSwish(nn.Module):
    """Memory Efficient Swish.

    Args:
        nn (module): torch.nn

    Returns:
        [type]: [description]
    """

    class F(torch.autograd.Function):
        """F.

        Args:
            torch ([type]): [description]

        Returns:
            [type]: [description]
        """

        @staticmethod
        def forward(ctx, x):
            """Forward Propagation.

            Args:
                ctx ([Type]): A special container object to save
                                   any information that may be needed for the call to backward.
                x (torch.Tensor): Tensor activated element-wise

            Returns:
                [type]: [description]
            """
            ctx.save_for_backward(x)
            return x * torch.sigmoid(x)

        @staticmethod
        def backward(ctx, grad_output):
            """Backward Propagation.

            Args:
                ctx ([type]): [description]
                grad_output ([type]): [description]

            Returns:
                [type]: [description]
            """
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            return grad_output * (sx * (1 + x * (1 - sx)))

    def forward(self, x):
        """Forward Propagation.

        Args:
            x (torch.Tensor): Tensor activated element-wise

        Returns:
            [type]: [description]
        """
        return self.F.apply(x)


class Mish(nn.Module):
    """Mish. Self Regularized Non-Monotonic Activation Function.

    https://github.com/digantamisra98/Mish

    Args:
        nn (module): torch.nn
    """

    @staticmethod
    def forward(x):
        """Forward Propagation.

        Args:
            x (torch.Tensor): Tensor activated element-wise

        Returns:
            [type]: [description]
        """
        return x * F.softplus(x).tanh()


class MemoryEfficientMish(nn.Module):
    """Memory Efficient Mish.

    Args:
        nn (module): torch.nn
    """

    class F(torch.autograd.Function):
        """[summary]

        Args:
            torch ([type]): [description]
        """

        @staticmethod
        def forward(ctx, x):
            """Forward Propagation.

            Args:
                ctx ([type]): [description]
                x (torch.Tensor): Tensor activated element-wise

            Returns:
                [type]: [description]
            """
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            """Backward Propagation.

            Args:
                ctx ([type]): [description]
                grad_output ([type]): [description]

            Returns:
                [type]: [description]
            """
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        """Forward Propagation.

        Args:
            x (torch.Tensor): Tensor activated element-wise

        Returns:
            [type]: [description]
        """
        return self.F.apply(x)


# FReLU https://arxiv.org/abs/2007.11824 -------------------------------------------------------------------------------
class FReLU(nn.Module):
    """Funnel Activation.

    https://arxiv.org/abs/2007.11824

    Args:
        nn (module): torch.nn
    """

    def __init__(self, c1, k=3):
        """Init.

        Args:
            c1 (torch.Tensor): ch_in
            k (int, optional): kernel. Defaults to 3.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        """Forward Propagation.

        Args:
            x (torch.Tensor): Tensor activated element-wise

        Returns:
            [type]: [description]
        """
        return torch.max(x, self.bn(self.conv(x)))
