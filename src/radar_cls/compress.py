"""INT8 post-training quantization with ONNX Runtime, and FP32 vs INT8 comparison.

    python -m radar_cls.compress --checkpoint runs/default/best.pt --calib-images 512

Steps: export FP32 ONNX -> static QDQ quantization (per-channel weights, calibration on train
images) -> accuracy on the test split and CPU latency for both models.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .checkpoint import load_model
from .config import config_from_dict
from .data import build_loaders
from .export import export_onnx
from .profiling import cpu_name
from .tracking import Tracker


def _require_ort():
    try:
        import onnxruntime as ort
        from onnxruntime import quantization as q
    except ImportError as exc:
        raise ImportError('INT8 quantization needs onnxruntime: pip install -e ".[onnx]"') from exc
    return ort, q


class _CalibReader:
    def __init__(self, loader, max_images: int):
        self.batches = []
        n = 0
        for images, _ in loader:
            for img in images:
                self.batches.append({"input": img.unsqueeze(0).numpy()})
                n += 1
                if n >= max_images:
                    break
            if n >= max_images:
                break
        self._it = iter(self.batches)

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        self._it = iter(self.batches)


def quantize_int8(fp32_path: Path, int8_path: Path, calib_loader, calib_images: int = 512) -> Path:
    _, q = _require_ort()
    pre = fp32_path.with_name(fp32_path.stem + ".pre.onnx")
    q.shape_inference.quant_pre_process(str(fp32_path), str(pre))
    q.quantize_static(
        str(pre),
        str(int8_path),
        _CalibReader(calib_loader, calib_images),
        quant_format=q.QuantFormat.QDQ,
        per_channel=True,
        activation_type=q.QuantType.QUInt8,
        weight_type=q.QuantType.QInt8,
        calibrate_method=q.CalibrationMethod.MinMax,
    )
    pre.unlink(missing_ok=True)
    return int8_path


def _session(path: Path, threads: int = 1):
    ort, _ = _require_ort()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def ort_accuracy(path: Path, loader) -> dict[str, float]:
    sess = _session(path, threads=0)  # all cores for the accuracy pass
    ys, ps = [], []
    for images, labels in loader:
        logits = sess.run(None, {"input": images.numpy()})[0]
        ps.append(logits.argmax(1))
        ys.append(labels.numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
    }


def ort_latency_ms(path: Path, image_size: int, threads: int = 1, warmup: int = 10, runs: int = 50) -> float:
    sess = _session(path, threads=threads)
    x = np.random.default_rng(0).standard_normal((1, 3, image_size, image_size), dtype=np.float32)
    for _ in range(warmup):
        sess.run(None, {"input": x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"input": x})
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def run(
    checkpoint: str,
    output_dir: str | None = None,
    calib_images: int = 512,
    split: str = "test",
    tracker: Tracker | None = None,
) -> dict:
    tracker = tracker or Tracker()
    model, ckpt = load_model(checkpoint, "cpu")
    cfg = config_from_dict(ckpt["config"])
    out = Path(output_dir or Path(checkpoint).parent / "compress")
    out.mkdir(parents=True, exist_ok=True)

    # Calibration uses validation images (deterministic eval transform), never the test split.
    bundle = build_loaders(cfg.data, cfg.augment, cfg.train.seed, splits=(split,))
    calib = build_loaders(cfg.data, cfg.augment, cfg.train.seed, splits=("val",)).loaders["val"]
    image_size = int(next(iter(bundle.loaders[split]))[0].shape[-1])

    fp32 = export_onnx(model, torch.randn(1, 3, image_size, image_size), out / "model_fp32.onnx")
    int8 = quantize_int8(fp32, out / "model_int8.onnx", calib, calib_images)

    result = {"split": split, "image_size": image_size, "cpu": cpu_name(), "calib_images": calib_images}
    for name, path in (("fp32", fp32), ("int8", int8)):
        acc = ort_accuracy(path, bundle.loaders[split])
        result[name] = {
            **acc,
            "size_mb": path.stat().st_size / 2**20,
            "cpu_latency_ms_1thread": ort_latency_ms(path, image_size, threads=1),
        }
    result["accuracy_drop_pp"] = 100 * (result["fp32"]["accuracy"] - result["int8"]["accuracy"])
    result["size_ratio"] = result["fp32"]["size_mb"] / result["int8"]["size_mb"]
    result["speedup"] = result["fp32"]["cpu_latency_ms_1thread"] / result["int8"]["cpu_latency_ms_1thread"]

    (out / "compress.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    tracker.summary(
        {
            f"int8/{split}_accuracy": result["int8"]["accuracy"],
            f"fp32_onnx/{split}_accuracy": result["fp32"]["accuracy"],
            "int8/accuracy_drop_pp": result["accuracy_drop_pp"],
            "int8/size_mb": result["int8"]["size_mb"],
            "int8/cpu_latency_ms": result["int8"]["cpu_latency_ms_1thread"],
            "int8/speedup": result["speedup"],
        }
    )
    print(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--calib-images", type=int, default=512)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args(argv)
    run(args.checkpoint, args.output_dir, args.calib_images, args.split)


if __name__ == "__main__":
    main()
