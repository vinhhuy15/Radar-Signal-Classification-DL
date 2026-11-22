"""End-to-end smoke test: train -> resume -> evaluate -> export -> predict, on CPU with tiny data."""

import json
from pathlib import Path

import pytest
import torch

from radar_cls import evaluate, export, predict, train
from radar_cls.checkpoint import load_model
from radar_cls.config import apply_overrides, load_config

from .conftest import CLASSES


def test_train_evaluate_export_predict(cfg, image_folder):
    summary = train.run(cfg)
    out = Path(cfg.output_dir)
    for name in ("best.pt", "last.pt", "config.yaml", "history.json", "train.log"):
        assert (out / name).exists(), name
    assert summary["best_epoch"] >= 1
    history = json.loads((out / "history.json").read_text())
    assert len(history) == cfg.train.epochs
    # LR logged for epoch 1 must be the configured initial LR (the old script printed the next one).
    assert history[0]["lr"] == pytest.approx(cfg.train.lr)

    metrics = evaluate.run(str(out / "best.pt"), split="test")
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(metrics["confusion_matrix"]) == len(CLASSES)

    paths = export.run(str(out / "best.pt"), image_size=32, onnx=False)
    ts, meta = export.load_torchscript(paths["torchscript"])
    assert meta["class_names"] == CLASSES
    model, _ = load_model(out / "best.pt")
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(ts(x), model(x))

    results = predict.run(str(paths["torchscript"]), [str(image_folder / "LFM")], top_k=2)
    assert results and all(r["top"][0]["class"] in CLASSES for r in results)


def test_resume_continues_training(cfg):
    cfg.train.epochs = 1
    train.run(cfg)
    cfg.train.epochs = 2
    train.run(cfg, resume=str(Path(cfg.output_dir) / "last.pt"))
    history = json.loads((Path(cfg.output_dir) / "history.json").read_text())
    assert [h["epoch"] for h in history] == [1, 2]


def test_parameter_budget_is_enforced(cfg):
    cfg.model.max_params = 1_000
    with pytest.raises(ValueError, match="budget"):
        train.run(cfg)


def test_onnx_export(cfg):
    pytest.importorskip("onnx")
    cfg.train.epochs = 1
    train.run(cfg)
    paths = export.run(str(Path(cfg.output_dir) / "best.pt"), image_size=32, onnx=True)
    import onnx

    onnx.checker.check_model(onnx.load(str(paths["onnx"])))


def test_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  lr: 0.1\n")
    cfg = load_config(p, ["train.lr=1e-3", "augment.hflip=true", "data.image_size=null"])
    assert cfg.train.lr == pytest.approx(1e-3)
    assert cfg.augment.hflip is True and cfg.data.image_size is None
    with pytest.raises(ValueError):
        load_config(None, ["train.not_a_key=1"])
    with pytest.raises(ValueError):
        apply_overrides({}, ["missing_equals"])


def test_default_config_file_loads():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "default.yaml")
    assert cfg.augment.hflip is False
    assert cfg.data.val_fraction + cfg.data.test_fraction < 1


def test_train_with_cache_and_device_normalization(cfg, tmp_path):
    # Same pipeline with the uint8 cache: batches are normalized in to_device(), not in workers.
    cfg.data.cache_dir = str(tmp_path / "cache")
    cfg.output_dir = str(tmp_path / "cached_run")
    summary = train.run(cfg)
    assert summary["best_epoch"] >= 1
    history = json.loads((Path(cfg.output_dir) / "history.json").read_text())
    assert all(0.0 <= row["val_accuracy"] <= 1.0 for row in history)
