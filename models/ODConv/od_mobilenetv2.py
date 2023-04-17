
import torch
from torch import nn
import numpy as np
from models.ODConv.odconv import ODConv2d

__all__ = ['od_mobilenetv2_050', 'od_mobilenetv2_075', 'od_mobilenetv2_100']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=nn.BatchNorm2d):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class ODConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=nn.BatchNorm2d,
                 reduction=0.0625, kernel_num=1):
        padding = (kernel_size - 1) // 2
        super(ODConvBNReLU, self).__init__(
            ODConv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups,
                     reduction=reduction, kernel_num=kernel_num),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=nn.BatchNorm2d, reduction=0.0625, kernel_num=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ODConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer,
                                       reduction=reduction, kernel_num=kernel_num))
        layers.extend([
            # dw
            ODConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer,
                         reduction=reduction, kernel_num=kernel_num),
            # pw-linear
            ODConv2d(hidden_dim, oup, 1, 1, 0,
                     reduction=reduction, kernel_num=kernel_num),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class OD_MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=InvertedResidual,
                 norm_layer=nn.BatchNorm2d,
                 dropout=0.2,
                 reduction=0.0625,
                 kernel_num=1,
                 **kwargs):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(OD_MobileNetV2, self).__init__()

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer,
                                      reduction=reduction, kernel_num=kernel_num))
                input_channel = output_channel
        # building last several layers
        features.append(ODConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer,
                                     reduction=reduction, kernel_num=kernel_num))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        self.channel = [i.size(1) for i in self.forward(torch.randn(2, 3, 640, 640))]
        
    def net_update_temperature(self, temperature):
        for m in self.modules():
            if hasattr(m, "update_temperature"):
                m.update_temperature(temperature)      

    def forward(self, x):
        input_size = x.size(2)
        scale = [4, 8, 16, 32]
        features = [None, None, None, None]
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if input_size // x.size(2) in scale:
                features[scale.index(input_size // x.size(2))] = x
        return features

def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}
    for k, v in weight_dict.items():
        if k.replace('module.', '') in model_dict.keys() and np.shape(model_dict[k.replace('module.', '')]) == np.shape(v):
            temp_dict[k.replace('module.', '')] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f'loading weights... {idx}/{len(model_dict)} items')
    return model_dict

def od_mobilenetv2_050(weights=None, kernel_num=1):
    model = OD_MobileNetV2(width_mult=0.5, kernel_num=kernel_num)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')['state_dict']
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model

def od_mobilenetv2_075(weights=None, kernel_num=1):
    model = OD_MobileNetV2(width_mult=0.75, kernel_num=kernel_num)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')['state_dict']
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model

def od_mobilenetv2_100(weights=None, kernel_num=1):
    model = OD_MobileNetV2(width_mult=1.0, kernel_num=kernel_num)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')['state_dict']
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model