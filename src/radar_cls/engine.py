"""Training/evaluation loops, EMA and MixUp."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


class ModelEMA:
    """Exponential moving average of model weights. BatchNorm buffers are tracked too."""

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: bool = False):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.warmup = warmup
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1 + self.updates) / (10 + self.updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.current_decay()
        self.updates += 1
        model_state = model.state_dict()
        for name, ema_v in self.ema.state_dict().items():
            model_v = model_state[name].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(d).add_(model_v, alpha=1.0 - d)
            else:
                ema_v.copy_(model_v)


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


@dataclass
class EpochStats:
    loss: float
    accuracy: float


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    ema: ModelEMA | None = None,
    mixup_alpha: float = 0.0,
    use_amp: bool = False,
) -> EpochStats:
    """One pass over ``loader``. Accuracy under MixUp is the lam-weighted match to both targets."""
    model.train()
    loss_sum, correct, total = 0.0, 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        inputs, y_a, y_b, lam = mixup_batch(images, labels, mixup_alpha)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(inputs)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        n = labels.size(0)
        loss_sum += loss.item() * n
        pred = outputs.argmax(1)
        correct += lam * (pred == y_a).sum().item() + (1 - lam) * (pred == y_b).sum().item()
        total += n
    return EpochStats(loss=loss_sum / total, accuracy=correct / total)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    """Return (labels, predictions, probabilities) as numpy arrays."""
    model.eval()
    ys, preds, probs = [], [], []
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True)).float()
        p = logits.softmax(1)
        ys.append(labels.numpy())
        preds.append(p.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_sum, total = 0.0, 0
    ys, preds = [], []
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images).float()
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        total += labels.size(0)
        ys.append(labels.cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(preds)
    return {
        "loss": loss_sum / total,
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
    }
