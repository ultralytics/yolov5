# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Activation functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        """
        Applies the Sigmoid-weighted Linear Unit (SiLU) activation function to the input tensor.

        Args:
            x (torch.Tensor): Input tensor to which the SiLU function will be applied.

        Returns:
            torch.Tensor: Output tensor with SiLU activation applied, maintaining the same shape as the input.

        See Also:
            https://arxiv.org/pdf/1606.08415.pdf for more details about the SiLU activation function.
        """
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):
    @staticmethod
    def forward(x):
        """
        Applies the Hardswish activation function, compatible with TorchScript, CoreML, and ONNX.

        Args:
            x (torch.Tensor): Input tensor to which Hardswish activation will be applied. Should be a float tensor.

        Returns:
            torch.Tensor: Tensor after applying the Hardswish activation function. Has the same shape as the input.

        Links:
            https://arxiv.org/abs/1905.02244

        Notes:
            Hardswish is equivalent to `x * F.hardsigmoid(x)` and is typically used to introduce non-linearity in neural network
            architectures, providing a computationally efficient approximation of the Swish activation function.
        """
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0  # for TorchScript, CoreML and ONNX


class Mish(nn.Module):
    """Mish activation https://github.com/digantamisra98/Mish."""

    @staticmethod
    def forward(x):
        """
        Applies the Mish activation function, a smooth alternative to ReLU.

        Args:
            x (torch.Tensor): Input tensor to which the Mish activation function will be applied.

        Returns:
            torch.Tensor: Output tensor after applying the Mish activation function.

        Notes:
            For more information, refer to the Mish repository:
            https://github.com/digantamisra98/Mish

        Example:
            ```python
            import torch
            from ultralytics import Mish

            input_tensor = torch.randn(3, 3)
            output_tensor = Mish.forward(input_tensor)
            ```
        """
        return x * F.softplus(x).tanh()


class MemoryEfficientMish(nn.Module):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            """
            Applies the Mish activation function, a smooth alternative to ReLU, to the input tensor `x`.

            Args:
                ctx (torch.autograd.function.FunctionCtx): The context object that can be used to stash information
                                                           for backward computation.
                x (torch.Tensor): The input tensor to which the Mish activation function is applied.

            Returns:
                torch.Tensor: The activated tensor after applying the Mish function.

            Notes:
                The Mish activation function is defined as x * tanh(softplus(x)). Softplus is a smooth approximation
                to the ReLU function, and tanh provides a non-linearity. This combination results in smoother and
                potentially more informative gradient flows compared to the traditional ReLU function,
                facilitating better learning dynamics in deep neural networks. For more details, refer to
                https://github.com/digantamisra98/Mish.
            """
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            """
            Computes the gradient of the Mish activation function with respect to input `x`.

            Args:
                ctx (torch.autograd.Function): Context object that can be used to stash information for backward computation.
                grad_output (torch.Tensor): Tensor of gradients of the loss with respect to the output of the
                    forward function.

            Returns:
                torch.Tensor: Gradient of the loss with respect to the input `x`.

            Notes:
                For more information on the Mish activation function, refer to
                https://github.com/digantamisra98/Mish.
            """
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        """
        Applies the Mish activation function to the input tensor `x`.

        Args:
            x (torch.Tensor): Input tensor on which the Mish activation function will be applied.

        Returns:
            torch.Tensor: The result of applying the Mish activation function to the input tensor.

        Notes:
            The Mish activation function is defined as `x * tanh(ln(1 + exp(x)))`.

        See Also:
            https://github.com/digantamisra98/Mish for more details on the Mish activation function.
        """
        return self.F.apply(x)


class FReLU(nn.Module):
    """FReLU activation https://arxiv.org/abs/2007.11824."""

    def __init__(self, c1, k=3):  # ch_in, kernel
        """
        Initializes the FReLU activation function with specified input channels and kernel size.

        Args:
            c1 (int): Number of input channels.
            k (int): Kernel size for the convolution operation. Default is 3.

        Returns:
            None: This is an initializer and does not return a value.

        Notes:
            This class implements the Funnel ReLU (FReLU) activation function, as described in
            "Funnel Activation for Visual Recognition" (https://arxiv.org/abs/2007.11824).
            The activation function integrates a depthwise convolution that helps capture
            local context better than traditional ReLU. Here's an example of using FReLU:

        Example:
            ```python
            import torch
            from ultralytics import FReLU

            x = torch.randn(1, 32, 224, 224)  # Example input tensor
            frelu = FReLU(32, 3)
            output = frelu(x)
            ```
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        """
        Applies the Funnel ReLU (FReLU) activation function using the max operation between the input tensor and the
        batch normalization (BN) convolved input.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) where N is the batch size, C is the number of channels,
                              H is the height, and W is the width.

        Returns:
            torch.Tensor: Output tensor after applying the FReLU activation, maintaining the same shape as the input.

        Notes:
            - The FReLU activation function, as described in https://arxiv.org/abs/2007.11824, is particularly beneficial
              for computer vision tasks by combining the strengths of ReLU and self-attention mechanisms.
        """
        return torch.max(x, self.bn(self.conv(x)))


class AconC(nn.Module):
    """
    ACON activation (activate or not) function.

    AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    See "Activate or Not: Learning Customized Activation" https://arxiv.org/pdf/2009.04759.pdf.
    """

    def __init__(self, c1):
        """
        Initializes the AconC activation function with learnable parameters for channel-wise activation control.

        Args:
            c1 (int): The number of input channels.

        Returns:
            None

        Example:
            ```python
            aconc = AconC(c1=64)
            output = aconc(input_tensor)
            ```

        Note:
            AconC: (p1*x - p2*x) * sigmoid(beta*(p1*x - p2*x)) + p2*x, where beta is a learnable parameter. See
            "Activate or Not: Learning Customized Activation" https://arxiv.org/pdf/2009.04759.pdf.
        """
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, c1, 1, 1))

    def forward(self, x):
        """
        Applies AconC activation function, implementing customized activation using learnable parameters.

        Args:
            x (torch.Tensor): Input tensor to which the AconC activation function is applied.

        Returns:
            torch.Tensor: Tensor with AconC activation applied.

        Notes:
            The AconC activation function is defined as:
            (p1*x - p2*x) * sigmoid(beta * (p1*x - p2*x)) + p2*x
            where p1, p2, and beta are learnable parameters.

            For more details, refer to the paper:
            "Activate or Not: Learning Customized Activation" https://arxiv.org/pdf/2009.04759.pdf.
        """
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(self.beta * dpx) + self.p2 * x


class MetaAconC(nn.Module):
    """
    ACON activation (activate or not) function.

    AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    See "Activate or Not: Learning Customized Activation" https://arxiv.org/pdf/2009.04759.pdf.
    """

    def __init__(self, c1, k=1, s=1, r=16):
        """
        Initializes MetaAconC with configurable parameters for advanced channel-wise activation control.

        Args:
            c1 (int): Number of input channels.
            k (int, optional): Kernel size for the intermediate convolution layer. Default is 1.
            s (int, optional): Stride for the intermediate convolution layer. Default is 1.
            r (int, optional): Reduction ratio for the intermediate convolution layer. Default is 16.

        Returns:
            None
        """
        super().__init__()
        c2 = max(r, c1 // r)
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.fc1 = nn.Conv2d(c1, c2, k, s, bias=True)
        self.fc2 = nn.Conv2d(c2, c1, k, s, bias=True)
        # self.bn1 = nn.BatchNorm2d(c2)
        # self.bn2 = nn.BatchNorm2d(c1)

    def forward(self, x):
        """
        Applies the Meta ACON activation function with learnable parameters for channel-wise control on input tensor
        `x`.

        Args:
            x (torch.Tensor): The input tensor on which to apply the activation function, of shape (N, C, H, W).

        Returns:
            torch.Tensor: Tensor with the Meta ACON activation applied, of the same shape as the input tensor (N, C, H, W).

        Notes:
            For detailed information, refer to "Activate or Not: Learning Customized Activation" (https://arxiv.org/pdf/2009.04759.pdf).

        Example usage:
            ```python
            import torch
            from ultyrolytics.models.common import MetaAconC

            x = torch.randn(1, 64, 32, 32)  # example input
            activation = MetaAconC(64)
            output = activation(x)
            ```
        """
        y = x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True)
        # batch-size 1 bug/instabilities https://github.com/ultralytics/yolov5/issues/2891
        # beta = torch.sigmoid(self.bn2(self.fc2(self.bn1(self.fc1(y)))))  # bug/unstable
        beta = torch.sigmoid(self.fc2(self.fc1(y)))  # bug patch BN layers removed
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(beta * dpx) + self.p2 * x
