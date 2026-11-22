"""Evaluate a checkpoint on one split and write metrics, a per-class report and plots.

    python -m radar_cls.evaluate --checkpoint runs/default/best.pt --split test

Run this on ``test`` once, after all model and hyperparameter choices are final.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .checkpoint import load_model
from .config import config_from_dict
from .data import build_loaders
from .engine import predict
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


def run(checkpoint: str, split: str = "test", output_dir: str | None = None, device: str = "auto") -> dict:
    dev = resolve_device(device)
    model, ckpt = load_model(checkpoint, dev)
    cfg = config_from_dict(ckpt["config"])
    out = Path(output_dir or Path(checkpoint).parent / f"eval_{split}")
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger(log_file=out / "eval.log")

    bundle = build_loaders(cfg.data, cfg.augment, cfg.train.seed, splits=(split,))
    if bundle.classes != list(ckpt["class_names"]):
        raise ValueError(f"Class mismatch between split files {bundle.classes} and checkpoint {ckpt['class_names']}")
    y, p, _ = predict(model, bundle.loaders[split], dev)
    metrics = compute_metrics(y, p, bundle.classes)
    metrics.update({"split": split, "checkpoint": str(checkpoint), "epoch": ckpt.get("epoch")})

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    try:
        plot_confusion_matrix(metrics["confusion_matrix"], bundle.classes, out / "confusion_matrix.png")
    except ImportError:
        log.info("matplotlib/seaborn not installed; skipping confusion matrix plot")

    log.info(
        f"[{split}] n={metrics['num_samples']} accuracy={metrics['accuracy']:.4f} macro-F1={metrics['macro_f1']:.4f}"
    )
    log.info("\n" + classification_report(y, p, target_names=bundle.classes, digits=4, zero_division=0))
    log.info(f"Results written to {out}")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    run(args.checkpoint, args.split, args.output_dir, args.device)


if __name__ == "__main__":
    main()
