"""Run an experiment suite on Kaggle or Colab with one command.

    python scripts/cloud_run.py <suite> [--gpus N] [extra args for radar_cls.experiments run]

What it does:
1. Finds (Kaggle) or downloads (Colab, via the Kaggle API) the dataset.
2. Reads WANDB_API_KEY (and on Colab KAGGLE_API_TOKEN, or KAGGLE_USERNAME / KAGGLE_KEY) from the platform's secrets.
3. Builds the split files and the uint8 image cache once.
4. Runs the suite on every visible GPU in parallel (one process per GPU, disjoint shards).
5. Runs CPU benchmarks (latency, INT8) alone afterwards and writes results/<suite>.md.

Finished runs are skipped and interrupted ones resume, so re-running continues the suite.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DATASET = "huynhthethien/radarcommunsignaldata2026train"
EXPECTED = {"LFM", "BPSK", "QPSK", "BARKER"}


def detect_platform() -> str:
    # Check Colab first: recent Colab runtimes also have a /kaggle/input folder.
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/input").exists():
        return "kaggle"
    if Path("/content").exists():
        return "colab"
    return "local"


def get_secret(platform: str, name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    try:
        if platform == "kaggle":
            from kaggle_secrets import UserSecretsClient

            return UserSecretsClient().get_secret(name)
        if platform == "colab":
            from google.colab import userdata

            return userdata.get(name)
    except Exception:  # noqa: BLE001 - a missing secret is not fatal
        return None
    return None


def find_dataset_root(base: Path) -> Path:
    """First folder under ``base`` whose subfolders look like our class names."""
    for dirpath, dirnames, _ in os.walk(base):
        names = {d.upper().replace("-", "").replace("_", "") for d in dirnames}
        if len(EXPECTED & names) >= 3:
            return Path(dirpath)
    raise FileNotFoundError(f"Could not find the dataset under {base}.")


def colab_download(work: Path) -> Path:
    target = Path("/content/data")
    if not target.exists() or not any(target.iterdir()):
        # Either a new-style API token (kaggle.com > Settings > API Tokens, starts with "KGAT_")
        # or the legacy kaggle.json pair. Accept the token under any of the usual names.
        names = ("KAGGLE_API_TOKEN", "KAGGLE_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY")
        found = {n: get_secret("colab", n) for n in names}
        print(f"[cloud_run] Kaggle secrets found: {[n for n, v in found.items() if v] or 'none'}", flush=True)
        token = found["KAGGLE_API_TOKEN"] or found["KAGGLE_TOKEN"]
        if not token:  # a new-style token saved under another name
            token = next((v for v in found.values() if v and v.strip().startswith("KGAT_")), None)
        if token:
            os.environ["KAGGLE_API_TOKEN"] = token.strip()
        elif found["KAGGLE_USERNAME"] and found["KAGGLE_KEY"]:
            os.environ["KAGGLE_USERNAME"] = found["KAGGLE_USERNAME"].strip()
            os.environ["KAGGLE_KEY"] = found["KAGGLE_KEY"].strip()
        else:
            raise RuntimeError(
                "Add KAGGLE_API_TOKEN (or KAGGLE_USERNAME + KAGGLE_KEY) to Colab Secrets "
                "(key icon on the left, with notebook access on) to download the dataset."
            )
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle"], check=True)
        target.mkdir(parents=True, exist_ok=True)
        print(f"[cloud_run] downloading {DATASET} to {target} ...", flush=True)
        subprocess.run(["kaggle", "datasets", "download", "-d", DATASET, "-p", str(target), "--unzip"], check=True)
    return find_dataset_root(target)


def resolve_paths(platform: str, runs_root: str | None) -> dict[str, Path]:
    if platform == "kaggle":
        work = Path("/kaggle/working")
        return {"work": work, "data": find_dataset_root(Path("/kaggle/input")), "runs": work / "runs"}
    if platform == "colab":
        work = Path("/content")
        # Runs default to Google Drive (if mounted) so they survive a runtime reset.
        drive = Path("/content/drive/MyDrive")
        runs = Path(runs_root) if runs_root else (drive / "radar_runs" if drive.exists() else work / "runs")
        return {"work": work, "data": colab_download(work), "runs": runs}
    work = Path.cwd()
    return {"work": work, "data": Path("data/radarcommunsignaldata2026train"), "runs": work / "runs"}


def write_base_config(paths: dict[str, Path], platform: str) -> Path:
    cfg = yaml.safe_load((REPO / "configs" / "kaggle.yaml").read_text())
    cfg["data"]["root"] = str(paths["data"])
    cfg["data"]["split_dir"] = str(paths["runs"] / "splits")
    cfg["data"]["cache_dir"] = "/tmp/radar_cache"
    cfg["wandb"]["tags"] = [platform]
    out = paths["work"] / f"{platform}_resolved.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", nargs="?", default="smoke")
    parser.add_argument("--gpus", type=int, default=None, help="parallel workers (default: all visible GPUs)")
    parser.add_argument("--runs-root", default=None)
    args, extra = parser.parse_known_args()

    platform = detect_platform()
    paths = resolve_paths(platform, args.runs_root)
    paths["runs"].mkdir(parents=True, exist_ok=True)
    print(f"[cloud_run] platform={platform} dataset={paths['data']} runs={paths['runs']}", flush=True)

    # Reuse committed split files so every platform evaluates on the same test images.
    (paths["runs"] / "splits").mkdir(parents=True, exist_ok=True)
    for f in (REPO / "splits").glob("*.csv"):
        target = paths["runs"] / "splits" / f.name
        if not target.exists():
            target.write_bytes(f.read_bytes())

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    key = get_secret(platform, "WANDB_API_KEY")
    if key:
        env["WANDB_API_KEY"] = key
    else:
        env["WANDB_MODE"] = "offline"
    print(f"[cloud_run] W&B: {'online' if key else 'offline (no WANDB_API_KEY secret)'}", flush=True)

    base = write_base_config(paths, platform)
    suite_file = str(REPO / "configs" / "experiments" / f"{args.suite}.yaml")
    common = ["--base", str(base), "--runs-root", str(paths["runs"])]
    exp = [sys.executable, "-m", "radar_cls.experiments"]

    print("[cloud_run] preparing splits and image cache (first time per session takes a few minutes)", flush=True)
    subprocess.run([*exp, "prepare", suite_file, *common], check=True, cwd=paths["work"], env=env)

    n = args.gpus if args.gpus is not None else max(1, gpu_count())
    procs = []
    for i in range(n):
        cmd = [*exp, "run", suite_file, *common, "--shard", f"{i}/{n}", *extra]
        if n > 1:
            cmd += ["--defer-cpu-benchmarks", "--set", "data.num_workers=2"]
        worker_env = {**env, "CUDA_VISIBLE_DEVICES": str(i)} if n > 1 else env
        print(f"[cloud_run] worker {i}: {' '.join(cmd)}", flush=True)
        procs.append(subprocess.Popen(cmd, cwd=paths["work"], env=worker_env))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"[cloud_run] a worker failed: exit codes {codes}")

    subprocess.run([*exp, "finalize", suite_file, *common], check=True, cwd=paths["work"], env=env)
    print(f"[cloud_run] done. Table: {paths['work'] / 'results' / (args.suite + '.md')}", flush=True)
    print(f"[cloud_run] split files to commit to the repo: {paths['runs'] / 'splits'}", flush=True)


if __name__ == "__main__":
    main()
