"""Baselines, ablation variants, cost metrics, robustness, INT8 and the suite runner."""

import json
from pathlib import Path

import pytest
import torch
import yaml

from radar_cls import compress, experiments
from radar_cls.config import config_from_dict
from radar_cls.models import available_models, build_model
from radar_cls.profiling import cost_report, count_flops
from radar_cls.utils import count_parameters


@pytest.mark.parametrize("name", ["plain_cnn", "mobilenetv3_small", "shufflenetv2", "resnet_tiny"])
def test_baselines_fit_the_budget(name):
    model = build_model(name, 12).eval()
    assert count_parameters(model) < 100_000
    assert model(torch.randn(2, 3, 224, 224)).shape == (2, 12)


@pytest.mark.parametrize(
    "kwargs", [{"attention": "none", "pooling": "avg"}, {"attention": "coord", "pooling": "avg"}, {"attention": "se"}]
)
def test_ablation_variants(kwargs):
    model = build_model("rf_spectronet", 12, **kwargs).eval()
    assert count_parameters(model) < 100_000
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 12)


def test_every_suite_variant_builds():
    root = Path(__file__).parents[1] / "configs" / "experiments"
    for f in root.glob("*.yaml"):
        suite = experiments.load_suite(f)
        for v in suite.variants:
            cfg = experiments.build_run_config(suite, v, 0, [], None)
            build_model(cfg.model.name, 12, dropout=cfg.model.dropout, **cfg.model.kwargs)
    assert {"rf_spectronet", "plain_cnn"} <= set(available_models())


def test_cost_report():
    model = build_model("rf_spectronet", 12)
    flops = count_flops(model, 224)
    assert 10e6 < flops < 1e9
    rep = cost_report(model, image_size=64, runs=3)
    assert rep["parameters"] == 86_196 and rep["cpu_latency_ms"] > 0


def _tiny_suite(tmp_path: Path, cfg, image_folder) -> Path:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "root": str(image_folder),
                    "split_dir": str(tmp_path / "splits"),
                    "batch_size": 8,
                    "num_workers": 0,
                },
                "train": {"epochs": 1, "device": "cpu", "tensorboard": False},
                "eval": {"noise_sigmas": [0.0, 0.5]},
            }
        )
    )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        yaml.safe_dump(
            {
                "name": "tiny",
                "base": str(base),
                "seeds": [0, 1],
                "runs_root": str(tmp_path / "runs"),
                "variants": {"full": [], "plain": ["model.name=plain_cnn"]},
                "compress": ["full"],
            }
        )
    )
    return suite


def test_suite_runs_resumes_and_aggregates(cfg, image_folder, tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    monkeypatch.chdir(tmp_path)
    suite = experiments.load_suite(_tiny_suite(tmp_path, cfg, image_folder))
    experiments.run_suite(suite)
    for v in ("full", "plain"):
        for s in (0, 1):
            d = tmp_path / "runs" / "tiny" / v / f"seed{s}"
            m = json.loads((d / "eval_test" / "metrics.json").read_text())
            assert [r["sigma"] for r in m["noise_robustness"]] == [0.0, 0.5]
            assert m["cost"]["parameters"] > 0
    assert (tmp_path / "runs" / "tiny" / "full" / "seed0" / "compress" / "compress.json").exists()
    # Second call skips finished runs (would be slow otherwise) and still aggregates.
    mtime = (tmp_path / "runs" / "tiny" / "full" / "seed0" / "best.pt").stat().st_mtime
    experiments.run_suite(suite)
    assert (tmp_path / "runs" / "tiny" / "full" / "seed0" / "best.pt").stat().st_mtime == mtime
    summary = json.loads((tmp_path / "results" / "tiny_summary.json").read_text())
    assert [s["variant"] for s in summary] == ["full", "plain"]
    assert all(s["n_seeds"] == 2 for s in summary)
    md = (tmp_path / "results" / "tiny.md").read_text()
    assert "± " in md and "full" in md


def test_int8_quantization(cfg):
    pytest.importorskip("onnxruntime")
    from radar_cls import train

    cfg.train.epochs = 1
    cfg.eval.measure_cost = False
    train.run(cfg)
    res = compress.run(str(Path(cfg.output_dir) / "best.pt"), calib_images=16)
    assert res["int8"]["size_mb"] < res["fp32"]["size_mb"]
    assert 0 <= res["int8"]["accuracy"] <= 1


def test_wandb_offline_logging(cfg, tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    from radar_cls import train

    monkeypatch.setenv("WANDB_SILENT", "true")
    cfg.train.epochs = 1
    cfg.eval.measure_cost = False
    cfg.wandb.enabled = True
    cfg.wandb.mode = "offline"
    cfg.wandb.log_model = True
    summary = train.run(cfg)
    out = Path(cfg.output_dir)
    assert (out / "wandb_run_id.txt").exists()
    assert list((out / "wandb").glob("offline-run-*")), "offline W&B run directory missing"
    assert "test_accuracy" in summary


def test_disabled_tracking_has_no_side_effects(cfg):
    cfg2 = config_from_dict(cfg.to_dict())
    assert cfg2.wandb.enabled is False


def test_sharded_suite_then_finalize(cfg, image_folder, tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    monkeypatch.chdir(tmp_path)
    suite = experiments.load_suite(_tiny_suite(tmp_path, cfg, image_folder))
    experiments.prepare(suite)
    experiments.run_suite(suite, shard=(0, 2), defer_cpu_benchmarks=True)
    done = sorted(p.parent.relative_to(tmp_path / "runs") for p in (tmp_path / "runs").rglob("train_summary.json"))
    assert len(done) == 2  # half of the 4 (seed, variant) pairs
    experiments.run_suite(suite, shard=(1, 2), defer_cpu_benchmarks=True)
    assert len(list((tmp_path / "runs").rglob("train_summary.json"))) == 4
    assert not list((tmp_path / "runs").rglob("compress.json")), "compress must wait for finalize"
    experiments.finalize(suite)
    costs = json.loads((tmp_path / "runs" / "tiny" / "costs.json").read_text())
    assert set(costs) == {"full", "plain"}
    assert len(list((tmp_path / "runs" / "tiny" / "full").rglob("compress.json"))) == 2
    summary = json.loads((tmp_path / "results" / "tiny_summary.json").read_text())
    assert all("cpu_latency_ms_mean" in s for s in summary)
