# Activation functions

import torch
import torch.nn as nn
import torch.nn.functional as F


# SiLU https://arxiv.org/pdf/1905.02244.pdf ----------------------------------------------------------------------------
class SiLU(nn.Module):
    """Export-friendly version of nn.SiLU().

    Args:
        nn (module): torch.nn
    """

    @staticmethod
    def forward(x):
        """[summary]

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
        """[summary]

        Args:
            x (torch.Tensor): Tensor activated element-wise

        Returns:
            [type]: [description]
        """
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX


class MemoryEfficientSwish(nn.Module):
    """[summary]

    Args:
        nn (module): torch.nn

    Returns:
        [type]: [description]
    """
    class F(torch.autograd.Function):
        """[summary]

        Args:
            torch ([type]): [description]

        Returns:
            [type]: [description]
        """

        @staticmethod
        def forward(ctx, x):
            """[summary]

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
            """[summary]

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
        """[summary]

        Args:
            x (torch.Tensor): Tensor activated element-wise

        Returns:
            [type]: [description]
        """
        return self.F.apply(x)

class Mish(nn.Module):
    """Mish. 
    
    https://github.com/digantamisra98/Mis

    Args:
        nn (module): torch.nn

    Returns:
        [type]: [description]
    """

    @staticmethod
    def forward(x):
        """[summary]

        Args:
            x (torch.Tensor): Tensor activated element-wise

        Returns:
            [type]: [description]
        """
        return x * F.softplus(x).tanh()


class MemoryEfficientMish(nn.Module):
    """[summary]

    Args:
        nn (module): torch.nn

    Returns:
        [type]: [description]
    """
    class F(torch.autograd.Function):
        """[summary]

        Args:
            torch ([type]): [description]

        Returns:
            [type]: [description]
        """

        @staticmethod
        def forward(ctx, x):
            """[summary]

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
            """[summary]

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
        """[summary]

        Args:
            x (torch.Tensor): Tensor activated element-wise

        Returns:
            [type]: [description]
        """
        return self.F.apply(x)


class FReLU(nn.Module):
    """Funnel Activation.

    https://arxiv.org/abs/2007.11824

    Args:
        nn (module): torch.nn
    """

    def __init__(self, c1, k=3):
        """[summary]

        Args:
            c1 (torch.Tensor): ch_in
            k (int, optional): kernel. Defaults to 3.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        """[summary]

        Args:
            x (torch.Tensor): Tensor activated element-wise

        Returns:
            [type]: [description]
        """
        return torch.max(x, self.bn(self.conv(x)))
