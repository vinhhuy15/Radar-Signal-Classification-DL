"""Model registry. Add baselines here so they can be selected with ``model.name`` in a config."""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from .rf_spectronet import BasicCNN, CoordAtt, MBConvCA, RFSpectroNet

_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "rf_spectronet": RFSpectroNet,
}


def register_model(name: str):
    def deco(fn: Callable[..., nn.Module]):
        if name in _REGISTRY:
            raise KeyError(f"Model {name!r} is already registered")
        _REGISTRY[name] = fn
        return fn

    return deco


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def build_model(name: str, num_classes: int, **kwargs) -> nn.Module:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {available_models()}")
    return _REGISTRY[name](num_classes=num_classes, **kwargs)


__all__ = ["BasicCNN", "CoordAtt", "MBConvCA", "RFSpectroNet", "available_models", "build_model", "register_model"]
