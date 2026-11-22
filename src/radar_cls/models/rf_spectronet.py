"""RF-SpectroNet: MBConv blocks with Coordinate Attention and hybrid pooling (< 100k params).

The architecture is unchanged from the course version (``legacy/Main.py``, class ``BasicCNN``),
so weights trained with the old script load into this module.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CoordAtt(nn.Module):
    """Coordinate Attention (Hou et al., CVPR 2021)."""

    def __init__(self, inp: int, reduction: int = 32):
        super().__init__()
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        # Mean over W / H. Same as AdaptiveAvgPool2d((None, 1)) / ((1, None)) but ONNX-exportable.
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w.permute(0, 1, 3, 2)))
        return x * a_w * a_h


class MBConvCA(nn.Module):
    """Mobile inverted bottleneck with Coordinate Attention on the expanded features."""

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: float):
        super().__init__()
        self.use_res_connect = stride == 1 and inp == oup
        hidden_dim = int(round(inp * expand_ratio))
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(inp, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim), nn.SiLU(inplace=True)]
        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        ]
        self.conv = nn.Sequential(*layers)
        self.ca = CoordAtt(hidden_dim)
        self.project = nn.Sequential(nn.Conv2d(hidden_dim, oup, 1, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.project(self.ca(self.conv(x)))
        return x + out if self.use_res_connect else out


class RFSpectroNet(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            MBConvCA(16, 24, stride=2, expand_ratio=2),
            MBConvCA(24, 40, stride=2, expand_ratio=2),
            MBConvCA(40, 64, stride=2, expand_ratio=2),
            MBConvCA(64, 80, stride=2, expand_ratio=2),
            MBConvCA(80, 128, stride=1, expand_ratio=2),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128 * 2, num_classes)  # avg + max pooled features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.stem(x))
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1).flatten(1)
        return self.fc(self.dropout(x))


# Backwards-compatible name used in the report and the legacy script.
BasicCNN = RFSpectroNet
