"""Export a checkpoint to TorchScript and (optionally) ONNX, with preprocessing metadata.

    python -m radar_cls.export --checkpoint runs/default/best.pt --image-size 224 --onnx

The TorchScript file embeds ``metadata.json`` (class names, normalization, input size), so a
consumer needs nothing else to run it. Export always happens on CPU so the file loads anywhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .checkpoint import load_model
from .config import config_from_dict

METADATA_KEY = "metadata.json"


def build_metadata(ckpt: dict, image_size: int) -> dict:
    cfg = config_from_dict(ckpt["config"])
    return {
        "model": cfg.model.name,
        "class_names": list(ckpt["class_names"]),
        "input_size": [3, image_size, image_size],
        "mean": list(cfg.data.mean),
        "std": list(cfg.data.std),
        "resize": cfg.data.image_size,
        "epoch": ckpt.get("epoch"),
        "val_metrics": ckpt.get("val_metrics"),
    }


def export_torchscript(model: torch.nn.Module, example: torch.Tensor, metadata: dict, path: Path) -> Path:
    traced = torch.jit.trace(model, example)
    torch.jit.save(traced, str(path), _extra_files={METADATA_KEY: json.dumps(metadata)})
    return path


def load_torchscript(path: str | Path, device: str = "cpu") -> tuple[torch.jit.ScriptModule, dict]:
    extra = {METADATA_KEY: ""}
    module = torch.jit.load(str(path), map_location=device, _extra_files=extra)
    return module.eval(), json.loads(extra[METADATA_KEY])


def export_onnx(model: torch.nn.Module, example: torch.Tensor, path: Path, opset: int = 17) -> Path:
    torch.onnx.export(
        model,
        example,
        str(path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    return path


def run(checkpoint: str, image_size: int = 224, output_dir: str | None = None, onnx: bool = False) -> dict[str, Path]:
    model, ckpt = load_model(checkpoint, "cpu")
    out = Path(output_dir or Path(checkpoint).parent / "export")
    out.mkdir(parents=True, exist_ok=True)
    example = torch.randn(1, 3, image_size, image_size)
    metadata = build_metadata(ckpt, image_size)

    paths = {"torchscript": export_torchscript(model, example, metadata, out / "model.torchscript.pt")}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    paths["metadata"] = out / "metadata.json"
    if onnx:
        paths["onnx"] = export_onnx(model, example, out / "model.onnx")
    for kind, p in paths.items():
        print(f"{kind}: {p}")
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--onnx", action="store_true", help="also export ONNX (needs the onnx package)")
    args = parser.parse_args(argv)
    run(args.checkpoint, args.image_size, args.output_dir, args.onnx)


if __name__ == "__main__":
    main()
