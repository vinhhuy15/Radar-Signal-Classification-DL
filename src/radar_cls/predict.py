"""Classify spectrogram images with an exported TorchScript model.

python -m radar_cls.predict --model runs/default/export/model.torchscript.pt path/to/img.png [more ...]
python -m radar_cls.predict --model runs/default/export/model.torchscript.pt path/to/folder --top-k 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets.folder import IMG_EXTENSIONS

from .export import load_torchscript


def preprocess_from_metadata(meta: dict) -> transforms.Compose:
    ops = []
    if meta.get("resize"):
        ops.append(transforms.Resize((meta["resize"], meta["resize"])))
    ops += [transforms.ToTensor(), transforms.Normalize(meta["mean"], meta["std"])]
    return transforms.Compose(ops)


def collect_images(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTENSIONS)
        else:
            files.append(p)
    return files


@torch.no_grad()
def run(model_path: str, inputs: list[str], top_k: int = 1, device: str = "cpu") -> list[dict]:
    model, meta = load_torchscript(model_path, device)
    tf = preprocess_from_metadata(meta)
    names = meta["class_names"]
    results = []
    for f in collect_images(inputs):
        with Image.open(f) as img:
            x = tf(img.convert("RGB")).unsqueeze(0).to(device)
        prob = model(x).softmax(1)[0]
        values, idx = prob.topk(min(top_k, len(names)))
        results.append(
            {
                "file": str(f),
                "top": [
                    {"class": names[i], "prob": round(float(v), 4)}
                    for v, i in zip(values.tolist(), idx.tolist(), strict=True)
                ],
            }
        )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="TorchScript file from radar_cls.export")
    parser.add_argument("inputs", nargs="+", help="image files or folders")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    for r in run(args.model, args.inputs, args.top_k, args.device):
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
