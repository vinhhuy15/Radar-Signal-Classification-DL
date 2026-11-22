"""Dataset splitting, transforms and data loaders.

Splits are stratified per class and written to CSV files so that every run, every seed and
every baseline uses exactly the same train/val/test images.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
def build_transforms(
    data: DataConfig, augment: AugmentConfig, train: bool, from_cache: bool = False
) -> transforms.Compose:
    """Train transform may be random; eval transform is always deterministic.

    ``from_cache``: inputs are already-resized uint8 CHW tensors from the memmap cache.
    """
    ops: list = []
    if data.image_size is not None and not from_cache:
        ops.append(transforms.Resize((data.image_size, data.image_size)))
    if train:
        if augment.hflip:
            ops.append(transforms.RandomHorizontalFlip())
        if augment.color_jitter > 0:
            ops.append(transforms.ColorJitter(brightness=augment.color_jitter, contrast=augment.color_jitter))
    if from_cache:
        # Jitter/flip on uint8 tensors match the PIL path; then scale to [0, 1] like ToTensor.
        ops += [transforms.ConvertImageDtype(torch.float32), transforms.Normalize(data.mean, data.std)]
    else:
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


class CachedDataset(Dataset):
    """Images decoded once into ``<cache_dir>/<key>.u8`` (N, 3, H, W uint8, memory-mapped)."""

    def __init__(self, images: np.ndarray, labels: list[int], transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = torch.from_numpy(np.array(self.images[idx]))  # copy out of the memmap
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def _cache_key(data: DataConfig, rows: list[tuple[str, int]]) -> str:
    h = hashlib.sha1()
    h.update(f"{Path(data.root).resolve()}|{data.image_size}|".encode())
    for rel, label in rows:
        h.update(f"{rel}:{label};".encode())
    return h.hexdigest()[:16]


def load_or_build_cache(data: DataConfig, rows: list[tuple[str, int]], name: str) -> np.ndarray:
    """Return a read-only memmap of all images in ``rows``; build it on first use."""
    cache = Path(data.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = _cache_key(data, rows)
    path, meta = cache / f"{name}_{key}.u8", cache / f"{name}_{key}.json"
    if path.exists() and meta.exists():
        shape = tuple(json.loads(meta.read_text())["shape"])
        return np.memmap(path, dtype=np.uint8, mode="r", shape=shape)

    ops = [transforms.PILToTensor()]
    if data.image_size is not None:
        ops.insert(0, transforms.Resize((data.image_size, data.image_size)))
    ds = SplitDataset(data.root, rows, transforms.Compose(ops))
    first = ds[0][0]
    shape = (len(ds), *first.shape)
    tmp = path.with_suffix(".tmp")
    arr = np.memmap(tmp, dtype=np.uint8, mode="w+", shape=shape)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=data.num_workers)
    i = 0
    for images, _ in loader:
        if images.shape[1:] != first.shape:
            raise ValueError(
                f"All images must have the same size to cache them; set data.image_size. Got {images.shape}"
            )
        arr[i : i + len(images)] = images.numpy()
        i += len(images)
    arr.flush()
    del arr
    tmp.replace(path)
    meta.write_text(json.dumps({"shape": list(shape), "root": str(data.root), "image_size": data.image_size}))
    return np.memmap(path, dtype=np.uint8, mode="r", shape=shape)


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
        if data.cache_dir:
            images = load_or_build_cache(data, rows[s], s)
            tf = build_transforms(data, augment, train=is_train, from_cache=True)
            ds = CachedDataset(images, [y for _, y in rows[s]], tf)
        else:
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
