"""Optional Weights & Biases tracking. Everything is a no-op when W&B is disabled or not installed."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import Config


class Tracker:
    def __init__(self, run: Any = None):
        self.run = run

    @property
    def active(self) -> bool:
        return self.run is not None

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(data, step=step)

    def summary(self, data: dict[str, Any]) -> None:
        if self.run is not None:
            for k, v in data.items():
                self.run.summary[k] = v

    def log_confusion_matrix(self, y_true, y_pred, class_names: list[str], key: str) -> None:
        if self.run is None:
            return
        import wandb

        self.run.log(
            {
                key: wandb.plot.confusion_matrix(
                    y_true=list(map(int, y_true)), preds=list(map(int, y_pred)), class_names=class_names
                )
            }
        )

    def log_table(self, key: str, columns: list[str], rows: list[list[Any]]) -> None:
        if self.run is None:
            return
        import wandb

        self.run.log({key: wandb.Table(columns=columns, data=rows)})

    def log_image(self, key: str, path: str | Path) -> None:
        if self.run is not None and Path(path).exists():
            import wandb

            self.run.log({key: wandb.Image(str(path))})

    def log_artifact(self, path: str | Path, name: str, kind: str = "model", metadata: dict | None = None) -> None:
        if self.run is None:
            return
        import wandb

        art = wandb.Artifact(name=name, type=kind, metadata=metadata or {})
        art.add_file(str(path))
        self.run.log_artifact(art)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None


def start_tracking(cfg: Config, output_dir: Path, resume_id: str | None = None) -> Tracker:
    w = cfg.wandb
    if not w.enabled or w.mode == "disabled":
        return Tracker()
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb.enabled is true but wandb is not installed: pip install wandb") from exc
    run_id_file = output_dir / "wandb_run_id.txt"
    if resume_id is None and run_id_file.exists():
        resume_id = run_id_file.read_text().strip()  # resuming a run continues the same W&B run
    run = wandb.init(
        project=w.project,
        entity=w.entity,
        group=w.group,
        name=w.name or output_dir.name,
        tags=list(w.tags),
        mode=w.mode,
        config=cfg.to_dict(),
        dir=str(output_dir),
        id=resume_id,
        resume="allow" if resume_id else None,
        settings=wandb.Settings(silent=True) if os.environ.get("RADAR_CLS_QUIET_WANDB") else None,
    )
    run_id_file.write_text(run.id)
    return Tracker(run)
