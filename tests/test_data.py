from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from radar_cls.config import AugmentConfig, DataConfig
from radar_cls.data import build_transforms, prepare_splits, scan_image_folder

from .conftest import CLASSES, PER_CLASS


def test_splits_are_disjoint_complete_and_stratified(cfg):
    classes, splits = prepare_splits(cfg.data)
    assert classes == CLASSES
    paths = {s: {p for p, _ in rows} for s, rows in splits.items()}
    assert (
        not (paths["train"] & paths["val"])
        and not (paths["train"] & paths["test"])
        and not (paths["val"] & paths["test"])
    )
    assert sum(len(v) for v in paths.values()) == len(CLASSES) * PER_CLASS
    for rows in splits.values():
        counts = Counter(y for _, y in rows)
        assert set(counts) == set(range(len(CLASSES))), "every class must appear in every split"


def test_splits_are_saved_and_reloaded_identically(cfg):
    _, first = prepare_splits(cfg.data)
    for s in ("train", "val", "test"):
        assert (Path(cfg.data.split_dir) / f"{s}.csv").exists()
    _, second = prepare_splits(cfg.data)
    assert first == second


def test_split_is_reproducible_for_same_seed(image_folder, tmp_path):
    a = prepare_splits(DataConfig(root=str(image_folder), split_dir=str(tmp_path / "a")))[1]
    b = prepare_splits(DataConfig(root=str(image_folder), split_dir=str(tmp_path / "b")))[1]
    assert a == b


def test_scan_ignores_non_images(image_folder):
    (image_folder / "LFM" / "notes.txt").write_text("not an image")
    _, samples = scan_image_folder(image_folder)
    assert all(not p.endswith(".txt") for p, _ in samples)


def test_eval_transform_is_deterministic(image_folder):
    img = Image.open(next((image_folder / "BPSK").glob("*.png"))).convert("RGB")
    tf = build_transforms(DataConfig(), AugmentConfig(hflip=True, color_jitter=0.5), train=False)
    assert torch.equal(tf(img), tf(img))
    assert not any(isinstance(t, (transforms.RandomHorizontalFlip, transforms.ColorJitter)) for t in tf.transforms)


def test_no_flip_by_default():
    tf = build_transforms(DataConfig(), AugmentConfig(), train=True)
    assert not any(isinstance(t, transforms.RandomHorizontalFlip) for t in tf.transforms)
    tf = build_transforms(DataConfig(), AugmentConfig(hflip=True), train=True)
    assert any(isinstance(t, transforms.RandomHorizontalFlip) for t in tf.transforms)
