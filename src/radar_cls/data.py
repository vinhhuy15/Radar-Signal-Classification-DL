"""Dataset splitting, transforms and data loaders.

Splits are stratified per class and written to CSV files so that every run, every seed and
every baseline uses exactly the same train/val/test images.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import IMG_EXTENSIONS

from .config import AugmentConfig, DataConfig
from .utils import seed_worker

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- transforms
def build_transforms(data: DataConfig, augment: AugmentConfig, train: bool) -> transforms.Compose:
    """Train transform may be random; eval transform is always deterministic."""
    ops: list = []
    if data.image_size is not None:
        ops.append(transforms.Resize((data.image_size, data.image_size)))
    if train:
        if augment.hflip:
            ops.append(transforms.RandomHorizontalFlip())
        if augment.color_jitter > 0:
            ops.append(transforms.ColorJitter(brightness=augment.color_jitter, contrast=augment.color_jitter))
    ops += [transforms.ToTensor(), transforms.Normalize(data.mean, data.std)]
    return transforms.Compose(ops)


# --------------------------------------------------------------------------- splitting
def scan_image_folder(root: str | Path) -> tuple[list[str], list[tuple[str, int]]]:
    """Return (class_names, [(relative_path, label), ...]) for an ImageFolder-style tree."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not classes:
        raise ValueError(f"No class folders found in {root}")
    samples = []
    for label, cls in enumerate(classes):
        for path in sorted((root / cls).rglob("*")):
            if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS:
                samples.append((path.relative_to(root).as_posix(), label))
    if not samples:
        raise ValueError(f"No images found in {root}")
    return classes, samples


def make_splits(
    samples: list[tuple[str, int]], val_fraction: float, test_fraction: float, seed: int
) -> dict[str, list[tuple[str, int]]]:
    """Stratified train/val/test split."""
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be in (0, 1) and sum to < 1")
    labels = [y for _, y in samples]
    rest, test = train_test_split(samples, test_size=test_fraction, stratify=labels, random_state=seed)
    rest_labels = [y for _, y in rest]
    val_size = val_fraction / (1.0 - test_fraction)
    train, val = train_test_split(rest, test_size=val_size, stratify=rest_labels, random_state=seed)
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def write_split_csv(path: Path, rows: list[tuple[str, int]], classes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "label", "class_name"])
        for rel, label in rows:
            writer.writerow([rel, label, classes[label]])


def read_split_csv(path: Path) -> list[tuple[str, int, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [(r["path"], int(r["label"]), r["class_name"]) for r in csv.DictReader(fh)]


def prepare_splits(data: DataConfig) -> tuple[list[str], dict[str, list[tuple[str, int]]]]:
    """Load the split CSVs, creating them from ``data.root`` the first time."""
    split_dir = Path(data.split_dir)
    files = {s: split_dir / f"{s}.csv" for s in SPLITS}
    if all(f.exists() for f in files.values()):
        loaded = {s: read_split_csv(f) for s, f in files.items()}
        names: dict[int, str] = {}
        for rows in loaded.values():
            for _, label, name in rows:
                names[label] = name
        classes = [names[i] for i in range(len(names))]
        return classes, {s: [(p, y) for p, y, _ in rows] for s, rows in loaded.items()}

    classes, samples = scan_image_folder(data.root)
    splits = make_splits(samples, data.val_fraction, data.test_fraction, data.split_seed)
    for s in SPLITS:
        write_split_csv(files[s], splits[s], classes)
    return classes, splits


# --------------------------------------------------------------------------- dataset
class SplitDataset(Dataset):
    def __init__(self, root: str | Path, rows: list[tuple[str, int]], transform=None):
        self.root = Path(root)
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        rel, label = self.rows[idx]
        with Image.open(self.root / rel) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


@dataclass
class DataBundle:
    classes: list[str]
    loaders: dict[str, DataLoader]


def build_loaders(data: DataConfig, augment: AugmentConfig, seed: int, splits=SPLITS) -> DataBundle:
    classes, rows = prepare_splits(data)
    pin = torch.cuda.is_available()
    loaders = {}
    for s in splits:
        is_train = s == "train"
        ds = SplitDataset(data.root, rows[s], build_transforms(data, augment, train=is_train))
        gen = torch.Generator().manual_seed(seed)
        loaders[s] = DataLoader(
            ds,
            batch_size=data.batch_size,
            shuffle=is_train,
            drop_last=is_train and len(ds) > data.batch_size,  # avoid a size-1 BatchNorm batch
            num_workers=data.num_workers,
            pin_memory=pin,
            worker_init_fn=seed_worker,
            generator=gen,
            persistent_workers=data.num_workers > 0,
        )
    return DataBundle(classes=classes, loaders=loaders)
