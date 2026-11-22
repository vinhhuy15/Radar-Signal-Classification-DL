"""Run an experiment suite (variants x seeds) and aggregate the results.

    python -m radar_cls.experiments run configs/experiments/ablation.yaml
    python -m radar_cls.experiments run configs/experiments/baselines.yaml --seeds 0 1 --set train.epochs=30
    python -m radar_cls.experiments aggregate configs/experiments/ablation.yaml
    # On Kaggle: use the Kaggle paths and keep runs in /kaggle/working
    python -m radar_cls.experiments run configs/experiments/ablation.yaml --base configs/kaggle.yaml \
        --runs-root /kaggle/working/runs

Each run writes to ``<runs_root>/<suite>/<variant>/seed<k>/``. A run that already has a
``train_summary.json`` is skipped, and one with a ``last.pt`` is resumed, so a suite can be
continued across several Kaggle sessions.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import apply_overrides, config_from_dict
from .utils import get_logger


@dataclass
class Suite:
    name: str
    base: str
    seeds: list[int]
    variants: dict[str, list[str]]
    common: list[str] = field(default_factory=list)
    compress: list[str] = field(default_factory=list)  # variants to INT8-quantize after training
    runs_root: str = "runs"


def load_suite(path: str | Path) -> Suite:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["variants"] = {k: list(v or []) for k, v in raw["variants"].items()}
    return Suite(**raw)


def run_dir(suite: Suite, variant: str, seed: int) -> Path:
    return Path(suite.runs_root) / suite.name / variant / f"seed{seed}"


def build_run_config(suite: Suite, variant: str, seed: int, extra: list[str], wandb: bool | None):
    with open(suite.base, encoding="utf-8") as fh:
        values = yaml.safe_load(fh) or {}
    overrides = [
        *suite.common,
        *suite.variants[variant],
        f"train.seed={seed}",
        f"output_dir={run_dir(suite, variant, seed).as_posix()}",
        f"wandb.group={suite.name}",
        f"wandb.name={variant}-s{seed}",
        f"wandb.tags=[{suite.name}, {variant}]",
        *extra,
    ]
    if wandb is not None:
        overrides.append(f"wandb.enabled={str(wandb).lower()}")
    return config_from_dict(apply_overrides(values, overrides))


def run_suite(
    suite: Suite,
    seeds: list[int] | None = None,
    only: list[str] | None = None,
    extra: list[str] | None = None,
    wandb: bool | None = None,
    shard: tuple[int, int] = (0, 1),
    defer_cpu_benchmarks: bool = False,
) -> None:
    """Run every (seed, variant) pair. ``shard=(i, n)`` runs only pairs with index % n == i,
    so n processes (e.g. one per GPU) can split a suite without overlapping.

    ``defer_cpu_benchmarks`` skips CPU latency and INT8 here (parallel workers would disturb the
    timings); run ``finalize`` afterwards, alone, to measure them."""
    from . import compress, train

    log = get_logger()
    variants = [v for v in suite.variants if not only or v in only]
    unknown = set(only or []) - set(suite.variants)
    if unknown:
        raise KeyError(f"Unknown variants {sorted(unknown)}; suite has {list(suite.variants)}")
    pairs = [(seed, variant) for seed in (seeds or suite.seeds) for variant in variants]
    index, count = shard
    for seed, variant in pairs[index::count]:
        out = run_dir(suite, variant, seed)
        if (out / "train_summary.json").exists():
            log.info(f"[skip] {variant} seed {seed}: already done")
            continue
        cfg = build_run_config(suite, variant, seed, extra or [], wandb)
        if defer_cpu_benchmarks:
            cfg.eval.measure_cost = False
        resume = out / "last.pt"
        log.info(f"[run] {suite.name}/{variant} seed {seed} -> {out}")
        train.run(cfg, resume=str(resume) if resume.exists() else None)
        if variant in suite.compress and not defer_cpu_benchmarks:
            try:
                compress.run(str(out / "best.pt"))
            except ImportError as exc:
                log.info(f"[compress skipped] {exc}")
    aggregate(suite)


def prepare(suite: Suite, extra: list[str] | None = None) -> None:
    """Create the split files and the image cache once, before parallel workers start."""
    from .data import SPLITS, build_loaders

    first = next(iter(suite.variants))
    cfg = build_run_config(suite, first, suite.seeds[0], extra or [], wandb=False)
    build_loaders(cfg.data, cfg.augment, cfg.train.seed, splits=SPLITS)


def finalize(suite: Suite) -> None:
    """CPU benchmarks run alone: per-variant cost (architecture only, so once per variant) and
    INT8 compression for the variants listed under ``compress``. Then aggregate."""
    from . import compress
    from .checkpoint import load_model
    from .profiling import cost_report

    log = get_logger()
    costs = {}
    for variant in suite.variants:
        runs = sorted((Path(suite.runs_root) / suite.name / variant).glob("seed*/best.pt"))
        if not runs:
            continue
        model, _ = load_model(runs[0], "cpu")
        image_size = 224
        metrics_file = runs[0].parent / "eval_test" / "metrics.json"
        if metrics_file.exists():
            image_size = json.loads(metrics_file.read_text()).get("cost", {}).get("image_size", 224)
        costs[variant] = cost_report(model, image_size=image_size)
        log.info(f"[cost] {variant}: {costs[variant]}")
        if variant in suite.compress:
            for ckpt in runs:
                if not (ckpt.parent / "compress" / "compress.json").exists():
                    try:
                        compress.run(str(ckpt))
                    except ImportError as exc:
                        log.info(f"[compress skipped] {exc}")
    cost_file = Path(suite.runs_root) / suite.name / "costs.json"
    cost_file.write_text(json.dumps(costs, indent=2), encoding="utf-8")
    aggregate(suite)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def collect(suite: Suite) -> list[dict[str, Any]]:
    rows = []
    cost_file = Path(suite.runs_root) / suite.name / "costs.json"
    suite_costs = json.loads(cost_file.read_text()) if cost_file.exists() else {}
    for variant in suite.variants:
        for seed_dir in sorted((Path(suite.runs_root) / suite.name / variant).glob("seed*")):
            metrics_file = seed_dir / "eval_test" / "metrics.json"
            if not metrics_file.exists():
                continue
            m = json.loads(metrics_file.read_text())
            row = {
                "variant": variant,
                "seed": int(seed_dir.name.removeprefix("seed")),
                "test_accuracy": m["accuracy"],
                "test_macro_f1": m["macro_f1"],
            }
            for r in m.get("noise_robustness", []):
                row[f"noise_acc_{r['sigma']:g}"] = r["accuracy"]
            cost = suite_costs.get(variant) or m.get("cost", {})
            for k in ("parameters", "flops", "cpu_latency_ms"):
                if k in cost:
                    row[k] = cost[k]
            comp = seed_dir / "compress" / "compress.json"
            if comp.exists():
                c = json.loads(comp.read_text())
                row.update(
                    {
                        "int8_accuracy": c["int8"]["accuracy"],
                        "int8_size_mb": c["int8"]["size_mb"],
                        "int8_latency_ms": c["int8"]["cpu_latency_ms_1thread"],
                        "fp32_onnx_latency_ms": c["fp32"]["cpu_latency_ms_1thread"],
                    }
                )
            rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]], variants: list[str]) -> list[dict[str, Any]]:
    out = []
    for v in variants:
        rs = [r for r in rows if r["variant"] == v]
        if not rs:
            continue
        s: dict[str, Any] = {"variant": v, "n_seeds": len(rs)}
        keys = sorted({k for r in rs for k in r} - {"variant", "seed"})
        for k in keys:
            vals = [r[k] for r in rs if k in r]
            s[f"{k}_mean"], s[f"{k}_std"] = _mean_std(vals)
        out.append(s)
    return out


def to_markdown(summary: list[dict[str, Any]]) -> str:
    def pct(s, k):
        m, sd = s.get(f"{k}_mean"), s.get(f"{k}_std")
        return "-" if m is None else f"{100 * m:.2f} ± {100 * sd:.2f}"

    noise_keys = sorted(
        {k[: -len("_mean")] for s in summary for k in s if k.startswith("noise_acc_") and k.endswith("_mean")},
        key=lambda k: float(k.removeprefix("noise_acc_")),
    )
    noise_keys = [k for k in noise_keys if k != "noise_acc_0"][-1:]  # report the hardest level
    header = ["Variant", "Seeds", "Params", "MFLOPs", "CPU ms", "Test acc (%)", "Test macro-F1 (%)"]
    header += [f"Acc @ noise σ={k.removeprefix('noise_acc_')} (%)" for k in noise_keys]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for s in summary:
        cells = [
            s["variant"],
            str(s["n_seeds"]),
            f"{s['parameters_mean']:,.0f}" if "parameters_mean" in s else "-",
            f"{s['flops_mean'] / 1e6:.1f}" if "flops_mean" in s else "-",
            f"{s['cpu_latency_ms_mean']:.2f}" if "cpu_latency_ms_mean" in s else "-",
            pct(s, "test_accuracy"),
            pct(s, "test_macro_f1"),
        ] + [pct(s, k) for k in noise_keys]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def aggregate(suite: Suite, results_dir: str | Path = "results") -> list[dict[str, Any]]:
    rows = collect(suite)
    summary = summarize(rows, list(suite.variants))
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{suite.name}_runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / f"{suite.name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = to_markdown(summary)
    (out / f"{suite.name}.md").write_text(md, encoding="utf-8")
    print(md)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("suite")
    r.add_argument("--seeds", type=int, nargs="*")
    r.add_argument("--only", nargs="*", help="run only these variants")
    r.add_argument(
        "--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE", help="extra overrides for every run"
    )
    r.add_argument("--wandb", choices=["on", "off"], default=None)
    r.add_argument("--shard", default="0/1", help="i/n: run only every n-th pair starting at i")
    r.add_argument("--defer-cpu-benchmarks", action="store_true", help="skip latency/INT8; run finalize later")
    fz = sub.add_parser("finalize", help="CPU latency per variant + INT8, then aggregate")
    fz.add_argument("suite")
    pr = sub.add_parser("prepare", help="build split files and image cache")
    pr.add_argument("suite")
    pr.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE")
    a = sub.add_parser("aggregate")
    a.add_argument("suite")
    a.add_argument("--results-dir", default="results")
    for p in (r, a, pr, fz):
        p.add_argument("--base", default=None, help="replace the suite's base config (e.g. configs/kaggle.yaml)")
        p.add_argument("--runs-root", default=None, help="replace the suite's runs_root")
    args = parser.parse_args(argv)
    suite = load_suite(args.suite)
    if args.base:
        suite.base = args.base
    if args.runs_root:
        suite.runs_root = args.runs_root
    if args.cmd == "run":
        wandb = None if args.wandb is None else args.wandb == "on"
        i, n = (int(x) for x in args.shard.split("/"))
        run_suite(suite, args.seeds, args.only, args.set, wandb, (i, n), args.defer_cpu_benchmarks)
    elif args.cmd == "finalize":
        finalize(suite)
    elif args.cmd == "prepare":
        prepare(suite, args.set)
    else:
        aggregate(suite, args.results_dir)


if __name__ == "__main__":
    main()
