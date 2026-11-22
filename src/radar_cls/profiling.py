"""Cost metrics: parameters, FLOPs (MACs x 2) and CPU latency.

Latency is measured on CPU with batch size 1, after warm-up, and reported as the median over
repeated runs. Always state the CPU model and thread count next to these numbers.
"""

from __future__ import annotations

import platform
import statistics
import time

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from .utils import count_parameters


def count_flops(model: nn.Module, image_size: int = 224) -> int:
    """Forward FLOPs for one image (a fused multiply-add counts as 2 FLOPs)."""
    model = model.eval()
    device = next(model.parameters()).device
    x = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(x)
    return int(counter.get_total_flops())


@torch.no_grad()
def cpu_latency_ms(
    model: nn.Module, image_size: int = 224, threads: int = 1, warmup: int = 10, runs: int = 50
) -> dict[str, float]:
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        model = model.to("cpu").eval()
        x = torch.randn(1, 3, image_size, image_size)
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    finally:
        torch.set_num_threads(previous)
    return {"median_ms": statistics.median(times), "p90_ms": sorted(times)[int(0.9 * (len(times) - 1))]}


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def model_size_mb(model: nn.Module) -> float:
    n_bytes = sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))
    return n_bytes / 2**20


def cost_report(model: nn.Module, image_size: int = 224, threads: int = 1, runs: int = 50) -> dict:
    import copy

    cpu_model = copy.deepcopy(model).to("cpu").eval()
    latency = cpu_latency_ms(cpu_model, image_size, threads=threads, runs=runs)
    return {
        "parameters": count_parameters(cpu_model),
        "flops": count_flops(cpu_model, image_size),
        "size_mb_fp32": model_size_mb(cpu_model),
        "cpu_latency_ms": latency["median_ms"],
        "cpu_latency_p90_ms": latency["p90_ms"],
        "cpu_threads": threads,
        "cpu": cpu_name(),
        "image_size": image_size,
    }
