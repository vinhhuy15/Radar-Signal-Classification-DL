"""Train a model.

    python -m radar_cls.train --config configs/default.yaml
    python -m radar_cls.train --config configs/default.yaml --set train.seed=1 output_dir=runs/seed1
    python -m radar_cls.train --config configs/default.yaml --resume runs/default/last.pt

Model selection uses the validation split only. The test split is never touched here;
run ``radar_cls.evaluate`` on it once, at the end.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .checkpoint import load_checkpoint, save_checkpoint
from .config import Config, load_config, save_config
from .data import build_loaders
from .engine import ModelEMA, evaluate, train_one_epoch
from .models import build_model
from .utils import count_parameters, get_logger, resolve_device, set_seed


def _tensorboard_writer(enabled: bool, log_dir: Path):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None
    return SummaryWriter(log_dir=str(log_dir))


def run(cfg: Config, resume: str | None = None) -> dict:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger(log_file=out / "train.log")
    save_config(cfg, out / "config.yaml")

    set_seed(cfg.train.seed, cfg.train.deterministic)
    device = resolve_device(cfg.train.device)
    use_amp = cfg.train.amp and device.type == "cuda"
    log.info(f"Device: {device} | AMP: {use_amp} | output: {out}")

    bundle = build_loaders(cfg.data, cfg.augment, cfg.train.seed, splits=("train", "val"))
    classes = bundle.classes
    log.info(
        f"Classes ({len(classes)}): {classes} | train={len(bundle.loaders['train'].dataset)} "
        f"val={len(bundle.loaders['val'].dataset)}"
    )

    model = build_model(cfg.model.name, num_classes=len(classes), dropout=cfg.model.dropout).to(device)
    n_params = count_parameters(model)
    log.info(f"Model: {cfg.model.name} | parameters: {n_params:,} (budget {cfg.model.max_params:,})")
    if n_params > cfg.model.max_params:
        raise ValueError(f"Model has {n_params:,} parameters, over the {cfg.model.max_params:,} budget")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.min_lr)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    ema = ModelEMA(model, cfg.train.ema_decay) if cfg.train.ema_decay > 0 else None

    start_epoch, best_score, best_epoch = 0, -1.0, -1
    history: list[dict] = []
    if resume:
        state = load_checkpoint(resume, device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        if ema is not None and state.get("ema") is not None:
            ema.ema.load_state_dict(state["ema"])
        start_epoch = state["epoch"] + 1
        best_score, best_epoch = state["best_score"], state["best_epoch"]
        history = state["history"]
        log.info(f"Resumed from {resume} at epoch {start_epoch + 1}")

    writer = _tensorboard_writer(cfg.train.tensorboard, out / "tb")
    metric = cfg.train.select_metric
    config_dict = cfg.to_dict()

    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]  # LR actually used in this epoch
        tr = train_one_epoch(
            model,
            bundle.loaders["train"],
            criterion,
            optimizer,
            device,
            scaler,
            ema=ema,
            mixup_alpha=cfg.augment.mixup_alpha,
            use_amp=use_amp,
        )
        eval_model = ema.ema if ema is not None else model
        val = evaluate(eval_model, bundle.loaders["val"], eval_criterion, device)
        scheduler.step()

        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": tr.loss,
            "train_acc": tr.accuracy,
            **{f"val_{k}": v for k, v in val.items()},
            "time_s": time.time() - t0,
        }
        history.append(row)
        log.info(
            f"Epoch {epoch + 1:03d}/{cfg.train.epochs} | lr {lr:.2e} | train loss {tr.loss:.4f} "
            f"acc {tr.accuracy:.4f} | val loss {val['loss']:.4f} acc {val['accuracy']:.4f} "
            f"F1 {val['macro_f1']:.4f} | {row['time_s']:.0f}s"
        )
        if writer is not None:
            for k, v in row.items():
                if k != "epoch":
                    writer.add_scalar(k, v, epoch + 1)

        if val[metric] > best_score:
            best_score, best_epoch = val[metric], epoch + 1
            save_checkpoint(
                out / "best.pt",
                state_dict=eval_model.state_dict(),
                class_names=classes,
                config=config_dict,
                epoch=epoch + 1,
                val_metrics=val,
            )
            log.info(f"  new best val {metric}: {best_score:.4f}")

        if (epoch + 1) % cfg.train.save_every == 0 or epoch + 1 == cfg.train.epochs:
            save_checkpoint(
                out / "last.pt",
                model=model.state_dict(),
                ema=ema.ema.state_dict() if ema is not None else None,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),
                epoch=epoch,
                best_score=best_score,
                best_epoch=best_epoch,
                history=history,
                class_names=classes,
                config=config_dict,
            )

    if writer is not None:
        writer.close()
    summary = {"best_epoch": best_epoch, f"best_val_{metric}": best_score, "parameters": n_params}
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(f"Done. Best val {metric} {best_score:.4f} at epoch {best_epoch}. Checkpoint: {out / 'best.pt'}")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    parser.add_argument("--resume", default=None, help="path to last.pt")
    args = parser.parse_args(argv)
    run(load_config(args.config, args.set), resume=args.resume)


if __name__ == "__main__":
    main()
