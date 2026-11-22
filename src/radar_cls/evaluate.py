"""Evaluate a checkpoint on one split and write metrics, a per-class report and plots.

    python -m radar_cls.evaluate --checkpoint runs/default/best.pt --split test

Run this on ``test`` once, after all model and hyperparameter choices are final.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .checkpoint import load_model
from .config import config_from_dict
from .data import build_loaders
from .engine import predict
from .profiling import cost_report
from .tracking import Tracker
from .utils import get_logger, resolve_device


def compute_metrics(y: np.ndarray, p: np.ndarray, class_names: list[str]) -> dict:
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", labels=labels, zero_division=0)),
        "per_class": classification_report(
            y, p, labels=labels, target_names=class_names, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y, p, labels=labels).tolist(),
        "num_samples": int(len(y)),
    }


def plot_confusion_matrix(cm: list[list[int]], class_names: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        np.array(cm), annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def noise_robustness(model, loader, device, sigmas, seed: int = 0) -> list[dict]:
    """Accuracy and macro-F1 with Gaussian noise of std ``sigma`` added to the normalized input.

    A proxy for channel noise (the dataset has no SNR labels). The noise is seeded, so every model
    sees exactly the same corrupted images.
    """
    model.eval()
    rows = []
    for sigma in sigmas:
        gen = torch.Generator().manual_seed(seed)
        ys, ps = [], []
        for images, labels in loader:
            if sigma > 0:
                images = images + sigma * torch.randn(images.shape, generator=gen)
            ps.append(model(images.to(device)).argmax(1).cpu().numpy())
            ys.append(labels.numpy())
        y, p = np.concatenate(ys), np.concatenate(ps)
        rows.append(
            {
                "sigma": float(sigma),
                "accuracy": float(accuracy_score(y, p)),
                "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            }
        )
    return rows


def plot_robustness(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["sigma"] for r in rows], [r["accuracy"] for r in rows], marker="o")
    ax.set_xlabel("Gaussian noise std (normalized input)")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(
    checkpoint: str,
    split: str = "test",
    output_dir: str | None = None,
    device: str = "auto",
    tracker: Tracker | None = None,
    robustness: bool = True,
    cost: bool | None = None,
) -> dict:
    dev = resolve_device(device)
    model, ckpt = load_model(checkpoint, dev)
    cfg = config_from_dict(ckpt["config"])
    out = Path(output_dir or Path(checkpoint).parent / f"eval_{split}")
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger(log_file=out / "eval.log")
    tracker = tracker or Tracker()

    bundle = build_loaders(cfg.data, cfg.augment, cfg.train.seed, splits=(split,))
    if bundle.classes != list(ckpt["class_names"]):
        raise ValueError(f"Class mismatch between split files {bundle.classes} and checkpoint {ckpt['class_names']}")
    loader = bundle.loaders[split]
    y, p, _ = predict(model, loader, dev)
    metrics = compute_metrics(y, p, bundle.classes)
    metrics.update({"split": split, "checkpoint": str(checkpoint), "epoch": ckpt.get("epoch")})

    if robustness and cfg.eval.noise_sigmas:
        metrics["noise_robustness"] = noise_robustness(model, loader, dev, cfg.eval.noise_sigmas, cfg.eval.noise_seed)
    if cost if cost is not None else cfg.eval.measure_cost:
        image_size = int(next(iter(loader))[0].shape[-1])
        metrics["cost"] = cost_report(model, image_size=image_size)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    try:
        plot_confusion_matrix(metrics["confusion_matrix"], bundle.classes, out / "confusion_matrix.png")
        if "noise_robustness" in metrics:
            plot_robustness(metrics["noise_robustness"], out / "noise_robustness.png")
    except ImportError:
        log.info("matplotlib/seaborn not installed; skipping plots")

    flat = {f"{split}/accuracy": metrics["accuracy"], f"{split}/macro_f1": metrics["macro_f1"]}
    for name, row in metrics["per_class"].items():
        if isinstance(row, dict) and name in bundle.classes:
            flat[f"{split}/f1/{name}"] = row["f1-score"]
    for row in metrics.get("noise_robustness", []):
        flat[f"{split}/noise_acc/sigma_{row['sigma']:g}"] = row["accuracy"]
    for k, v in metrics.get("cost", {}).items():
        if isinstance(v, (int, float)):
            flat[f"cost/{k}"] = v
    tracker.summary(flat)
    tracker.log_confusion_matrix(y, p, bundle.classes, key=f"{split}/confusion_matrix")
    if "noise_robustness" in metrics:
        tracker.log_table(
            f"{split}/noise_robustness",
            ["sigma", "accuracy", "macro_f1"],
            [[r["sigma"], r["accuracy"], r["macro_f1"]] for r in metrics["noise_robustness"]],
        )

    log.info(
        f"[{split}] n={metrics['num_samples']} accuracy={metrics['accuracy']:.4f} macro-F1={metrics['macro_f1']:.4f}"
    )
    log.info("\n" + classification_report(y, p, target_names=bundle.classes, digits=4, zero_division=0))
    for row in metrics.get("noise_robustness", []):
        log.info(f"  noise sigma={row['sigma']:<5g} accuracy={row['accuracy']:.4f}")
    if "cost" in metrics:
        c = metrics["cost"]
        log.info(
            f"  cost: {c['parameters']:,} params, {c['flops'] / 1e6:.1f} MFLOPs, "
            f"{c['cpu_latency_ms']:.2f} ms/img on {c['cpu']} ({c['cpu_threads']} thread)"
        )
    log.info(f"Results written to {out}")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-robustness", action="store_true", help="skip the noise robustness curve")
    parser.add_argument("--no-cost", action="store_true", help="skip FLOPs/latency measurement")
    args = parser.parse_args(argv)
    run(
        args.checkpoint,
        args.split,
        args.output_dir,
        args.device,
        robustness=not args.no_robustness,
        cost=False if args.no_cost else None,
    )


if __name__ == "__main__":
    main()
