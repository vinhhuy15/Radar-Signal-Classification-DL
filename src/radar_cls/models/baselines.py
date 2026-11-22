"""Baselines at a parameter budget comparable to RF-SpectroNet (~86k parameters for 12 classes).

- ``plain_cnn``: stride-2 3x3 conv + BN + SiLU stack, the start of the ablation chain.
- ``mobilenetv3_small``: torchvision MobileNetV3-Small, width 0.2 (~89k).
- ``shufflenetv2``: torchvision ShuffleNetV2 with reduced channels (~95k).
- ``plain_cnn`` defaults give ~86k and ``resnet_tiny`` ~89k parameters.
- ``resnet_tiny``: ResNet with BasicBlocks [1, 1, 1, 1] and reduced widths.

All are trained from scratch; no ImageNet weights, so the comparison is about architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tvm
from torchvision.models.resnet import BasicBlock

from .rf_spectronet import HybridPool


class PlainCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        widths: tuple[int, ...] = (16, 24, 32, 48, 56, 72),
        dropout: float = 0.3,
        pooling: str = "avg",
    ):
        super().__init__()
        layers: list[nn.Module] = []
        c = 3
        for i, w in enumerate(widths):
            stride = 2 if i < 5 else 1
            layers += [nn.Conv2d(c, w, 3, stride, 1, bias=False), nn.BatchNorm2d(w), nn.SiLU(inplace=True)]
            c = w
        self.features = nn.Sequential(*layers)
        self.pool = HybridPool(pooling)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c * self.pool.factor, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.pool(self.features(x))))


class ResNetTiny(nn.Module):
    def __init__(self, num_classes: int, widths: tuple[int, ...] = (16, 28, 40, 52), dropout: float = 0.3):
        super().__init__()
        c = widths[0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, 3, 2, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1)
        )
        stages = []
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2
            down = None
            if stride != 1 or c != w:
                down = nn.Sequential(nn.Conv2d(c, w, 1, stride, bias=False), nn.BatchNorm2d(w))
            stages.append(BasicBlock(c, w, stride=stride, downsample=down))
            c = w
        self.stages = nn.Sequential(*stages)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(self.stem(x)).mean(dim=(2, 3))
        return self.fc(self.dropout(x))


def mobilenetv3_small(num_classes: int, width_mult: float = 0.2, dropout: float = 0.2) -> nn.Module:
    return tvm.mobilenet_v3_small(weights=None, num_classes=num_classes, width_mult=width_mult, dropout=dropout)


def shufflenetv2(
    num_classes: int,
    stages_repeats: tuple[int, ...] = (4, 8, 4),
    stages_out_channels: tuple[int, ...] = (24, 32, 64, 128, 192),
    dropout: float = 0.0,
) -> nn.Module:
    del dropout  # torchvision ShuffleNetV2 has no dropout
    return tvm.ShuffleNetV2(list(stages_repeats), list(stages_out_channels), num_classes=num_classes)
