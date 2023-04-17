import torch
import torch.nn as nn
from models.ODConv.odconv import ODConv2d
import numpy as np

__all__ = ['od_resnet18', 'od_resnet34', 'od_resnet50', 'od_resnet101']


def odconv3x3(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                    reduction=reduction, kernel_num=kernel_num)


def odconv1x1(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,
                    reduction=reduction, kernel_num=kernel_num)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=0.0625, kernel_num=1):
        super(BasicBlock, self).__init__()
        self.conv1 = odconv3x3(inplanes, planes, stride, reduction=reduction, kernel_num=kernel_num)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = odconv3x3(planes, planes, reduction=reduction, kernel_num=kernel_num)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=0.0625, kernel_num=1):
        super(Bottleneck, self).__init__()
        self.conv1 = odconv1x1(inplanes, planes, reduction=reduction, kernel_num=kernel_num)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = odconv3x3(planes, planes, stride, reduction=reduction, kernel_num=kernel_num)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = odconv1x1(planes, planes * self.expansion, reduction=reduction, kernel_num=kernel_num)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class OD_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, dropout=0.1, reduction=0.0625, kernel_num=1):
        super(OD_ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], reduction=reduction, kernel_num=kernel_num)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, reduction=reduction, kernel_num=kernel_num)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, reduction=reduction, kernel_num=kernel_num)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, reduction=reduction, kernel_num=kernel_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        self.channel = [i.size(1) for i in self.forward(torch.randn(2, 3, 640, 640))]
        
    def net_update_temperature(self, temperature):
        for m in self.modules():
            if hasattr(m, "update_temperature"):
                m.update_temperature(temperature)

    def _make_layer(self, block, planes, blocks, stride=1, reduction=0.625, kernel_num=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, reduction=reduction, kernel_num=kernel_num))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, reduction=reduction, kernel_num=kernel_num))

        return nn.Sequential(*layers) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x = self.maxpool(x1)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        
        return [x1, x2, x3, x4, x5]

def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}
    for k, v in weight_dict.items():
        if k.replace('module.', '') in model_dict.keys() and np.shape(model_dict[k.replace('module.', '')]) == np.shape(v):
            temp_dict[k.replace('module.', '')] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f'loading weights... {idx}/{len(model_dict)} items')
    return model_dict

def od_resnet18(weights=None, kernel_num=1):
    model = OD_ResNet(BasicBlock, [2, 2, 2, 2], kernel_num=kernel_num)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')['state_dict']
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model

def od_resnet34(weights=None, kernel_num=1):
    model = OD_ResNet(BasicBlock, [3, 4, 6, 3], kernel_num=kernel_num)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')['state_dict']
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model

def od_resnet50(weights=None, kernel_num=1):
    model = OD_ResNet(Bottleneck, [3, 4, 6, 3], kernel_num=kernel_num)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')['state_dict']
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model

def od_resnet101(weights=None, kernel_num=1):
    model = OD_ResNet(Bottleneck, [3, 4, 23, 3], kernel_num=kernel_num)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')['state_dict']
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model