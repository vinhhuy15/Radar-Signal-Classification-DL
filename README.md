# RF Signal Spectrum Classification using a Lightweight CNN

[![CI](https://github.com/vinhhuy15/Radar-Signal-Classification-DL/actions/workflows/ci.yml/badge.svg)](https://github.com/vinhhuy15/Radar-Signal-Classification-DL/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Automatic recognition of **radio frequency (RF) signals** from spectrogram images with a CNN of
**fewer than 100,000 parameters**, aimed at edge devices.

The model, **RF-SpectroNet** (called `BasicCNN` in the report), stacks Mobile Inverted
Bottleneck (MBConv) blocks with **Coordinate Attention** and uses hybrid average + max pooling.
It has **86,196 parameters** for 12 classes.

---

## Classes

12 classes, spectrogram images at 224×224:

| Category | Signal classes |
|:---|:---|
| Communications | 16-QAM, B-FM, BPSK, CPFSK, DSB-AM, GFSK, PAM4, QPSK |
| Radar | Barker, LFM, Rect, Step-FM |

## Results

> **Status:** being re-measured under the evaluation protocol below. Numbers will be filled in
> from `metrics.json` of the test split, reported as mean ± std over several seeds.

| Model | Parameters | Test accuracy | Test macro-F1 |
|:---|:---:|:---:|:---:|
| RF-SpectroNet | 86,196 | _pending_ | _pending_ |

**Legacy result (not comparable).** The course version (`legacy/Main.py`) reported 91.83%
accuracy on a validation split that was also used to pick the best epoch, with random flip and
colour jitter applied to validation images, and with no separate test set. That number is kept
for history only.

## Evaluation protocol

- **Stratified train / val / test split (70 / 15 / 15)** with a fixed seed. The file lists are
  written to `splits/*.csv` on the first run; commit them so every experiment uses the same split.
- **Model selection on val only** (best epoch by macro-F1). The test split is evaluated once, at the end.
- **Deterministic eval transforms**: no augmentation on val/test.
- **No horizontal flip by default**: flipping a spectrogram reverses its time axis (for example it
  turns an up-chirp into a down-chirp). It can be enabled with `augment.hflip=true` as an ablation.

## Installation

```bash
git clone https://github.com/vinhhuy15/Radar-Signal-Classification-DL.git
cd Radar-Signal-Classification-DL
pip install -e .            # or: pip install -e ".[dev,onnx,tensorboard]"
```

## Data

Put the dataset in ImageFolder layout (one folder per class) and point `data.root` at it:

```
data/radarcommunsignaldata2026train/
├── 16QAM/  *.png
├── B-FM/   *.png
└── ...
```

On Kaggle, use `configs/kaggle.yaml`, which already points at the Kaggle input path.

## Usage

```bash
# Train (writes runs/default/{best.pt,last.pt,config.yaml,history.json,train.log,tb/})
python -m radar_cls.train --config configs/default.yaml

# Override any value without editing the file
python -m radar_cls.train --config configs/default.yaml --set train.seed=1 output_dir=runs/seed1

# Resume an interrupted run
python -m radar_cls.train --config configs/default.yaml --resume runs/default/last.pt

# Evaluate on the held-out test split (metrics.json, confusion_matrix.png)
python -m radar_cls.evaluate --checkpoint runs/default/best.pt --split test

# Export to TorchScript (+ ONNX) with class names and preprocessing embedded
python -m radar_cls.export --checkpoint runs/default/best.pt --image-size 224 --onnx

# Classify images or a folder with the exported model
python -m radar_cls.predict --model runs/default/export/model.torchscript.pt path/to/images --top-k 3
```

After `pip install -e .` the same commands are available as `radar-train`, `radar-evaluate`,
`radar-export` and `radar-predict`.

### Using the exported model in Python

```python
import torch
from PIL import Image
from radar_cls.export import load_torchscript
from radar_cls.predict import preprocess_from_metadata

model, meta = load_torchscript("runs/default/export/model.torchscript.pt")  # loads on CPU
x = preprocess_from_metadata(meta)(Image.open("sample.png").convert("RGB")).unsqueeze(0)
with torch.no_grad():
    probs = model(x).softmax(1)[0]
print(meta["class_names"][probs.argmax().item()], probs.max().item())
```

Preprocessing is `ToTensor()` followed by `Normalize(mean=0.5, std=0.5)` per channel. It is stored
in the model's metadata, so you do not need to hard-code it.

## Project structure

```
├── configs/
│   ├── default.yaml          # full model, clean protocol
│   └── kaggle.yaml           # Kaggle paths
├── src/radar_cls/
│   ├── config.py             # typed config, YAML loading, --set overrides
│   ├── data.py               # stratified splits, transforms, loaders
│   ├── models/               # model registry + RF-SpectroNet
│   ├── engine.py             # train/eval loops, EMA, MixUp
│   ├── train.py              # training CLI (checkpoints, resume, AMP, TensorBoard)
│   ├── evaluate.py           # metrics, per-class report, confusion matrix
│   ├── export.py             # TorchScript / ONNX export with metadata
│   └── predict.py            # inference CLI
├── tests/                    # pytest: model, data protocol, end-to-end smoke test
├── legacy/Main.py            # original course script, kept unchanged for reference
└── docs/Report.pdf           # course technical report
```

## Training recipe

| Component | Setting |
|:---|:---|
| Architecture | MBConv (expand 2) + Coordinate Attention, 5 blocks |
| Pooling | Global average + global max, concatenated |
| Regularization | MixUp (α = 0.2), label smoothing (0.1), dropout (0.3) |
| Optimizer | AdamW (lr = 2e-3, weight decay = 1e-2) |
| Schedule | Cosine annealing to 1e-6 over 80 epochs |
| Weight averaging | EMA (decay = 0.999), used for validation and the saved model |

## Development

```bash
pip install -e ".[dev]"
pytest            # CPU-only, about a minute, uses a tiny synthetic dataset
ruff check . && ruff format --check .
```

CI runs the same checks on every pull request.

## Limitations

- The dataset has no SNR labels, so robustness versus SNR is not measured yet.
- No measurements on embedded hardware; any latency figures will be from a desktop CPU.

## Author

Giang Vinh Huy, Ho Chi Minh City University of Technology and Education (HCMUTE).

## License

MIT, see [LICENSE](LICENSE).
