from __future__ import annotations

import torch
from torch import nn
from .block import C2f


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(
            out_planes, eps=1e-5, momentum=0.01, affine=True
        ) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        maximum = torch.max(x, dim=1, keepdim=True).values
        average = torch.mean(x, dim=1, keepdim=True)
        return torch.cat((maximum, average), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, 7, padding=3, relu=False)

    def forward(self, x):
        # Identity-preserving gate:
        # output awal mendekati x karena 2 * sigmoid(0) = 1.
        # Ini menjaga fitur pretrained YOLOv8s pada awal fine-tuning.
        scale = 2.0 * torch.sigmoid(
            self.conv(self.compress(x))
        )
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super().__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_cw = x.permute(0, 2, 1, 3).contiguous()
        x_cw = self.cw(x_cw)
        x_cw = x_cw.permute(0, 2, 1, 3).contiguous()

        x_hc = x.permute(0, 3, 2, 1).contiguous()
        x_hc = self.hc(x_hc)
        x_hc = x_hc.permute(0, 3, 2, 1).contiguous()

        if self.no_spatial:
            return (x_cw + x_hc) / 2.0

        x_hw = self.hw(x)
        return (x_cw + x_hc + x_hw) / 3.0


class C2fTriplet(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.attention = TripletAttention()

    def forward(self, x):
        return self.attention(super().forward(x))

    def forward_split(self, x):
        return self.attention(super().forward_split(x))
