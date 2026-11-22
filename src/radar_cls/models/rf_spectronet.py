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


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation, used as an alternative to Coordinate Attention in ablations."""

    def __init__(self, inp: int, reduction: int = 8):
        super().__init__()
        mip = max(8, inp // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(inp, mip, 1), nn.SiLU(inplace=True), nn.Conv2d(mip, inp, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


def make_attention(kind: str, channels: int) -> nn.Module:
    if kind == "coord":
        return CoordAtt(channels)
    if kind == "se":
        return SqueezeExcite(channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Unknown attention {kind!r}; use 'coord', 'se' or 'none'")


class HybridPool(nn.Module):
    """Global pooling head input: 'hybrid' concatenates average and max pooling, 'avg' uses average only."""

    def __init__(self, mode: str = "hybrid"):
        super().__init__()
        if mode not in ("hybrid", "avg", "max"):
            raise ValueError(f"Unknown pooling {mode!r}; use 'hybrid', 'avg' or 'max'")
        self.mode = mode
        self.factor = 2 if mode == "hybrid" else 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "avg":
            return x.mean(dim=(2, 3))
        if self.mode == "max":
            return x.amax(dim=(2, 3))
        return torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)


class MBConvCA(nn.Module):
    """Mobile inverted bottleneck with Coordinate Attention on the expanded features."""

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: float, attention: str = "coord"):
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
        self.ca = make_attention(attention, hidden_dim)
        self.project = nn.Sequential(nn.Conv2d(hidden_dim, oup, 1, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.project(self.ca(self.conv(x)))
        return x + out if self.use_res_connect else out


class RFSpectroNet(nn.Module):
    """Defaults reproduce the course model. ``attention`` and ``pooling`` exist for ablations."""

    def __init__(self, num_classes: int, dropout: float = 0.3, attention: str = "coord", pooling: str = "hybrid"):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            MBConvCA(16, 24, stride=2, expand_ratio=2, attention=attention),
            MBConvCA(24, 40, stride=2, expand_ratio=2, attention=attention),
            MBConvCA(40, 64, stride=2, expand_ratio=2, attention=attention),
            MBConvCA(64, 80, stride=2, expand_ratio=2, attention=attention),
            MBConvCA(80, 128, stride=1, expand_ratio=2, attention=attention),
        )
        self.pool = HybridPool(pooling)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128 * self.pool.factor, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.blocks(self.stem(x)))
        return self.fc(self.dropout(x))


# Backwards-compatible name used in the report and the legacy script.
BasicCNN = RFSpectroNet
