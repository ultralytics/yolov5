# This file contains modules common to various models

import torch
import torch.nn as nn
from models.common import Conv


class surrogate_focus(nn.Module):
    # surrogate_focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(surrogate_focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)

        with torch.no_grad():
            self.convsp = nn.Conv2d(3, 12, (2, 2), groups=1, bias=False, stride=(2, 2))
            self.convsp.weight.data = torch.zeros(self.convsp.weight.shape).float()
            for i in range(4):
                for j in range(3):
                    ch = i*3 + j
                    if ch>=0 and ch<3:
                        self.convsp.weight[ch:ch+1, j:j+1, 0, 0] = 1
                    elif ch>=3 and ch<6:
                        self.convsp.weight[ch:ch+1, j:j+1, 1, 0] = 1
                    elif ch>=6 and ch<9:
                        self.convsp.weight[ch:ch+1, j:j+1, 0, 1] = 1
                    elif ch>=9 and ch<12:
                        self.convsp.weight[ch:ch+1, j:j+1, 1, 1] = 1

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(self.convsp(x))