"""Entry point for Kaggle notebooks.

In a notebook cell (GPU + Internet on, dataset attached, secret WANDB_API_KEY added):

    !git clone -q -b <branch> https://github.com/vinhhuy15/Radar-Signal-Classification-DL /kaggle/working/repo
    !cd /kaggle/working/repo && pip install -q -e ".[onnx]" && python scripts/kaggle_run.py smoke

Suites: smoke, ablation, baselines (see configs/experiments/). Extra arguments are passed on to
``radar_cls.experiments run``, e.g. ``--seeds 0 --set train.epochs=30``.
Finished runs are skipped, so re-running the same cell in a new session continues the suite.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
EXPECTED = {"LFM", "BPSK", "QPSK", "Barker"}


def find_dataset_root() -> Path:
    """First folder under /kaggle/input whose subfolders look like our class names."""
    for dirpath, dirnames, _ in os.walk(INPUT):
        names = {d.upper().replace("-", "").replace("_", "") for d in dirnames}
        if len({e.upper() for e in EXPECTED} & names) >= 3:
            return Path(dirpath)
    raise FileNotFoundError("Could not find the dataset under /kaggle/input. Attach it to the notebook.")


def wandb_login() -> bool:
    try:
        from kaggle_secrets import UserSecretsClient

        key = UserSecretsClient().get_secret("WANDB_API_KEY")
    except Exception as exc:  # noqa: BLE001 - any failure means "no key"
        print(f"[kaggle_run] W&B key not available ({exc.__class__.__name__}); logging offline.")
        os.environ["WANDB_MODE"] = "offline"
        return False
    os.environ["WANDB_API_KEY"] = key
    return True


def write_config(root: Path) -> Path:
    cfg_path = REPO / "configs" / "kaggle.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["data"]["root"] = str(root)
    cfg["data"]["split_dir"] = str(WORK / "splits")
    out = WORK / "kaggle_resolved.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def main() -> None:
    suite = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    extra = sys.argv[2:]
    root = find_dataset_root()
    print(f"[kaggle_run] dataset: {root}")
    # Reuse committed split files if the repo has them, so Kaggle and local runs share one split.
    committed = REPO / "splits"
    (WORK / "splits").mkdir(parents=True, exist_ok=True)
    for f in committed.glob("*.csv"):
        target = WORK / "splits" / f.name
        if not target.exists():
            target.write_bytes(f.read_bytes())
    online = wandb_login()
    print(f"[kaggle_run] W&B: {'online' if online else 'offline'}")
    base = write_config(root)
    cmd = [
        sys.executable,
        "-m",
        "radar_cls.experiments",
        "run",
        str(REPO / "configs" / "experiments" / f"{suite}.yaml"),
        "--base",
        str(base),
        "--runs-root",
        str(WORK / "runs"),
        *extra,
    ]
    print("[kaggle_run]", " ".join(cmd))
    # Unbuffered so per-epoch logs show up in the notebook as they happen.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    subprocess.run(cmd, check=True, cwd=WORK, env=env)
    print(f"[kaggle_run] done. Results table: {WORK / 'results' / (suite + '.md')}")
    print(f"[kaggle_run] split files (commit these to the repo): {WORK / 'splits'}")


if __name__ == "__main__":
    main()
