import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn


# Swish ------------------------------------------------------------------------
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        return grad_output * (sx * (1 + x * (1 - sx)))


class MemoryEfficientSwish(nn.Module):
    @staticmethod
    def forward(x):
        return SwishImplementation.apply(x)


class HardSwish(nn.Module):  # https://arxiv.org/pdf/1905.02244.pdf
    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3, 0., 6., True) / 6.


class Swish(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


# Mish ------------------------------------------------------------------------
class MishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        fx = F.softplus(x).tanh()
        return grad_output * (fx + x * sx * (1 - fx * fx))


class MemoryEfficientMish(nn.Module):
    @staticmethod
    def forward(x):
        return MishImplementation.apply(x)


class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()
