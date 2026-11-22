"""Checkpoint format.

``best.pt``  : {"state_dict", "class_names", "config", "epoch", "val_metrics"} - EMA weights if EMA is on.
``last.pt``  : the above plus optimizer/scheduler/scaler/EMA/history so training can resume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import config_from_dict
from .models import build_model


def save_checkpoint(path: str | Path, **payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic: an interrupted save never corrupts the previous file


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=True)


def load_model(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the model described by a checkpoint and load its weights."""
    ckpt = load_checkpoint(path, device)
    cfg = config_from_dict(ckpt["config"])
    model = build_model(cfg.model.name, num_classes=len(ckpt["class_names"]), dropout=cfg.model.dropout)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
