from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from radar_cls.config import Config, config_from_dict

CLASSES = ["BPSK", "LFM", "QPSK"]
PER_CLASS = 20
SIZE = 32


@pytest.fixture(scope="session")
def image_folder(tmp_path_factory) -> Path:
    """Tiny synthetic ImageFolder: each class has a distinct bright band so it is learnable."""
    root = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    for label, cls in enumerate(CLASSES):
        (root / cls).mkdir()
        for i in range(PER_CLASS):
            img = rng.integers(0, 60, size=(SIZE, SIZE, 3), dtype=np.uint8)
            img[label * 10 : label * 10 + 8, :, :] = 230
            Image.fromarray(img).save(root / cls / f"{i:03d}.png")
    return root


@pytest.fixture
def cfg(image_folder: Path, tmp_path: Path) -> Config:
    return config_from_dict(
        {
            "output_dir": str(tmp_path / "run"),
            "data": {
                "root": str(image_folder),
                "split_dir": str(tmp_path / "splits"),
                "batch_size": 8,
                "num_workers": 0,
            },
            "train": {"epochs": 2, "device": "cpu", "tensorboard": False, "lr": 0.01},
        }
    )
