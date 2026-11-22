"""Experiment configuration: typed dataclasses loaded from YAML, with CLI overrides."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    # Root folder in ImageFolder layout: <root>/<class_name>/<image>.
    root: str = "data/radarcommunsignaldata2026train"
    # Folder holding train.csv / val.csv / test.csv. Created on first run if missing.
    split_dir: str = "splits"
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split_seed: int = 42
    # None keeps the native resolution. Set e.g. 224 to force a resize.
    image_size: int | None = None
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    batch_size: int = 32
    num_workers: int = 2


@dataclass
class AugmentConfig:
    # Horizontal flip reverses the time axis of a spectrogram (e.g. up-chirp -> down-chirp),
    # so it is off by default and should only be enabled as an explicit ablation.
    hflip: bool = False
    color_jitter: float = 0.2  # brightness/contrast strength; 0 disables
    mixup_alpha: float = 0.2  # 0 disables MixUp


@dataclass
class ModelConfig:
    name: str = "rf_spectronet"
    dropout: float = 0.3
    max_params: int = 100_000


@dataclass
class TrainConfig:
    epochs: int = 80
    lr: float = 2e-3
    weight_decay: float = 1e-2
    min_lr: float = 1e-6
    label_smoothing: float = 0.1
    ema_decay: float = 0.999  # 0 disables EMA
    amp: bool = True  # mixed precision, only used on CUDA
    select_metric: str = "macro_f1"  # "accuracy" or "macro_f1", measured on val
    seed: int = 42
    deterministic: bool = False  # slower; bit-exact cuDNN
    device: str = "auto"
    tensorboard: bool = True  # used only if tensorboard is installed
    save_every: int = 1  # write last.pt every N epochs


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "runs/default"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls: type, values: dict[str, Any]) -> Any:
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        default = getattr(cls(), name)
        if dataclasses.is_dataclass(default) and isinstance(value, dict):
            kwargs[name] = _build(type(default), value)
        elif isinstance(default, tuple) and isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def config_from_dict(values: dict[str, Any]) -> Config:
    return _build(Config, values or {})


def apply_overrides(values: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``a.b=value`` overrides; values are parsed as YAML (so 1e-3, true, null work)."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = values
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_value(raw)
    return values


def _parse_value(raw: str) -> Any:
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        # YAML 1.1 reads "1e-3" as a string; accept it as a float.
        try:
            return float(value)
        except ValueError:
            pass
    return value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    values: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            values = yaml.safe_load(fh) or {}
    values = apply_overrides(values, overrides or [])
    return config_from_dict(values)


def save_config(cfg: Config, path: str | Path) -> None:
    data = cfg.to_dict()
    # YAML-friendly: tuples become lists.
    for section in data.values():
        if isinstance(section, dict):
            for k, v in section.items():
                if isinstance(v, tuple):
                    section[k] = list(v)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
