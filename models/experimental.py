"""Experimental modules."""

import numpy as np
import torch
from torch import nn

from models.common import Conv, DWConv
from utils.google_utils import attempt_download


class CrossConv(nn.Module):
    """Cross Convolution Downsample.

    Args:
        nn (module): torch.nn
    """

    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        """Init.

        Args:
            c1 (any): ch_in
            c2 (any): ch_out
            k (int, optional): kernel. Defaults to 3.
            s (int, optional): stride. Defaults to 1.
            g (int, optional): groups. Defaults to 1.
            e (float, optional): expansion. Defaults to 1.0.
            shortcut (bool, optional): shortcut. Defaults to False.
        """
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward Propagation.

        Args:
            x (torch.Tensor): Tensor activated element-wise

        Returns:
            [type]: [description]
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):
    """Weighted sum of 2 or more layers.

     https://arxiv.org/abs/1911.09070

    Args:
        nn (module): torch.nn
    """

    def __init__(self, n, weight=False):  # n: number of inputs
        """Init.

        Args:
            n ([type]): [description]
            weight (bool, optional): [description]. Defaults to False.
        """
        super(Sum, self).__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1., n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        """Forward Propagation.

        Args:
            x (torch.Tensor): Tensor activated element-wise

        Returns:
            [type]: [description]
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


class GhostConv(nn.Module):
    """Ghost Convolution.

        https://github.com/huawei-noah/ghostnet

    Args:
        nn (module): torch.nn
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Init.

        Example:
            ch_in, ch_out, kernel, stride, groups

        Args:
            c1 (any): ch_in
            c2 (any): ch_out
            k (int, optional): kernel. Defaults to 1.
            s (int, optional): stride. Defaults to 1.
            g (int, optional): groups. Defaults to 1.
            act (bool, optional): [description]. Defaults to True.
        """
        super(GhostConv, self).__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        """Forward Propagation.

        Args:
            x (torch.Tensor): Tensor activated element-wise

        Returns:
            [type]: [description]
        """
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck.

        https://github.com/huawei-noah/ghostnet

    Args:
        nn (module): torch.nn
    """

    def __init__(self, c1, c2, k, s):
        """Init.

        Args:
            c1 ([type]): [description]
            c2 ([type]): [description]
            k ([type]): [description]
            s ([type]): [description]
        """
        super(GhostBottleneck, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Forward Propagation.

        Args:
            x (torch.Tensor): Tensor activated element-wise

        Returns:
            [type]: [description]
        """
        return self.conv(x) + self.shortcut(x)


class MixConv2d(nn.Module):
    """Mixed Depthwise Conv.

        https://arxiv.org/abs/1907.09595

    Args:
        nn (module): torch.nn
    """

    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        """Init.

        Args:
            c1 ([type]): [description]
            c2 ([type]): [description]
            k (tuple, optional): [description]. Defaults to (1, 3).
            s (int, optional): [description]. Defaults to 1.
            equal_ch (bool, optional): [description]. Defaults to True.
        """
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        """Forward Propagation.

        Args:
            x (torch.Tensor): Tensor activated element-wise

        Returns:
            [type]: [description]
        """
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    """Ensemble of models.

    Args:
        nn (module): torch.nn
    """

    def __init__(self):
        """Init."""
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        """Forward Propagation.

        Args:
            x (torch.Tensor): Tensor activated element-wise
            augment (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None):
    """Load an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a.

    Args:
        weights ([type]): [description]
        map_location ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        model.append(torch.load(w, map_location=map_location)['model'].float().fuse().eval())  # load FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble
