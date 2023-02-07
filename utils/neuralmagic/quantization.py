import torch

from models.common import Bottleneck, GhostBottleneck

try:
    from torch.nn.quantized import FloatFunctional
except Exception:
    FloatFunctional = None

__all__ = ["NMGhostBottleneck", "NMBottleneck", "update_model_bottlenecks"]


class _Add(torch.nn.Module):
    """
    Add node with quantization support
    """

    def __init__(self):
        super().__init__()

        if FloatFunctional:
            self.functional = FloatFunctional()
            self.wrap_qat = True
            self.qat_wrapper_kwargs = {
                "num_inputs": 2,
                "num_outputs": 0,
            }

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        if FloatFunctional:
            return self.functional.add(a, b)
        else:
            return torch.add(a, b)


class NMGhostBottleneck(GhostBottleneck):
    """
    Updated GhostBottleneck class with quantizable add nodes
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_node = _Add()

    def forward(self, x):
        return self.add_node(self.conv(x), self.shortcut(x))


class NMBottleneck(Bottleneck):
    """
    Updated Bottleneck class with quantizable add nodes
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_node = _Add()

    def forward(self, x):
        return (
            self.add_node(x, self.cv2(self.cv1(x)))
            if self.add
            else self.cv2(self.cv1(x))
        )


def update_model_bottlenecks(model: torch.nn.Module) -> torch.nn.Module:
    """
    Replace all found Bottleneck and GhostBottleneck modules in model with the Neural
    Magic updated ones, allowing for add module quantization
    """

    found_bottlenecks = []
    new_bottlenecks = []
    for name, param in model.named_modules():

        if isinstance(param, Bottleneck):
            found_bottlenecks.append(name.split("."))

            # Backwards deducing Bottleneck __init__ args. Refer to
            # models.common.Bottleneck for details
            channels_in = param.cv1.conv.in_channels
            channels_out = param.cv2.conv.out_channels
            channels_out_hidden = param.cv1.conv.out_channels
            updated_module = NMBottleneck(
                c1=channels_in,
                c2=channels_out,
                shortcut=param.add and channels_in == channels_out,
                g=param.cv2.conv.groups,
                e=channels_out_hidden / param.cv1.conv.out_channels,
            )

            # Non-strict dict loading because class is initialized with identity batch
            # norm and yolov5 default checkpoints can exclude batch norm
            updated_module.load_state_dict(param.state_dict(), strict=False)
            updated_module.cv1.bn.momentum = param.cv1.bn.momentum
            updated_module.cv1.bn.eps = param.cv1.bn.eps
            updated_module.cv2.bn.momentum = param.cv2.bn.momentum
            updated_module.cv2.bn.eps = param.cv2.bn.eps
            new_bottlenecks.append(updated_module)

        elif isinstance(param, GhostBottleneck):
            found_bottlenecks.append(name.split("."))

            # Backwards deducing GhostBottleneck __init__ args. Refer to
            # models.common.GhostBottleneck for details
            stride = 2 if not isinstance(param.conv[1], torch.nn.Identity) else 1
            kernel = param.conv[1].conv.kernel_size[0] if stride == 2 else 3
            updated_module = NMGhostBottleneck(
                c1=param.conv[0].cv1.conv.in_channels,
                c2=param.conv[0].cv1.conv.out_channels * 4,
                k=kernel,
                s=stride,
            )

            # Non-strict dict loading because class is initialized with identity batch
            # norm and yolov5 default checkpoints can exclude batch norm
            updated_module.load_state_dict(param.state_dict(), strict=False)
            updated_module.cv1.bn.momentum = param.cv1.bn.momentum
            updated_module.cv1.bn.eps = param.cv1.bn.eps
            updated_module.cv2.bn.momentum = param.cv2.bn.momentum
            updated_module.cv2.bn.eps = param.cv2.bn.eps
            new_bottlenecks.append(updated_module)

    # Replace found bottleneck modules
    for (*parent, k), new_module in zip(found_bottlenecks, new_bottlenecks):
        model.get_submodule(".".join(parent))[int(k)] = new_module

    return model
