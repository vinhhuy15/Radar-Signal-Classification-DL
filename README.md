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

## Experiments

Experiment suites live in `configs/experiments/`. Each suite is a list of variants (config
overrides) run over several seeds; results are aggregated as mean ± std on the test split.

| Suite | Question | Variants |
|:---|:---|:---|
| `ablation` | Which components earn their parameters? | A0 plain CNN → A1 +MBConv → A2 +Coordinate Attention → A3 +hybrid pooling → A4 +MixUp/label smoothing → A5 +EMA (full), plus SE-instead-of-CA and horizontal-flip side studies |
| `baselines` | How does it compare at the same budget? | RF-SpectroNet, MobileNetV3-Small (width 0.2), ShuffleNetV2 (reduced), ResNet-tiny, all < 100k params, trained from scratch |
| `smoke` | Does the pipeline work on the real data? | 2 epochs, 1 seed |

```bash
python -m radar_cls.experiments run configs/experiments/ablation.yaml
python -m radar_cls.experiments run configs/experiments/baselines.yaml --seeds 0 1 --set train.epochs=30
python -m radar_cls.experiments aggregate configs/experiments/ablation.yaml   # -> results/ablation.md
```

Finished runs are skipped and interrupted runs resume from `last.pt`, so a suite can be continued
across sessions. For every run the test evaluation also records:

- **Cost:** parameters, FLOPs, FP32 size and single-thread CPU latency (batch 1, median of 50 runs).
- **Noise robustness:** accuracy with Gaussian noise of increasing std added to the normalized
  input. This is a proxy: the dataset has no SNR labels, so a true accuracy-vs-SNR curve needs a
  generated dataset.
- **INT8 compression** (variants listed under `compress:`): ONNX Runtime static quantization
  (QDQ, per-channel, calibrated on val), with FP32 vs INT8 accuracy, size and latency.

### Weights & Biases

Set `wandb.enabled: true` (already on in `configs/kaggle.yaml`). Each run logs its config, per-epoch
train/val curves, test metrics, per-class F1, confusion matrix, the noise robustness table, cost
metrics and the best checkpoint as an artifact. Runs of a suite share a W&B group, and each run is
tagged with the suite and variant, which makes it easy to build the report tables and figures.

### Kaggle and Colab

`scripts/cloud_run.py` (called by `kaggle_run.py` / `colab_run.py`) finds or downloads the data,
reads secrets, builds the split files and a uint8 image cache once, runs the suite on every
visible GPU in parallel (one worker per GPU, disjoint shards), then measures CPU latency and INT8
alone and writes `results/<suite>.md`. Re-running continues where it stopped.

**Kaggle** (GPU T4 x2 + Internet on, dataset attached, secret `WANDB_API_KEY`):

```bash
!git clone -q https://github.com/vinhhuy15/Radar-Signal-Classification-DL /kaggle/working/repo
!cd /kaggle/working/repo && pip install -q -e ".[onnx]" && python scripts/kaggle_run.py ablation
```

**Colab** (secrets `WANDB_API_KEY`, `KAGGLE_USERNAME`, `KAGGLE_KEY`; mounting Drive keeps runs
across runtime resets). Ready-made notebook: `notebooks/colab_run.ipynb`.

```python
from google.colab import drive; drive.mount("/content/drive")
!git clone -q https://github.com/vinhhuy15/Radar-Signal-Classification-DL /content/repo
!cd /content/repo && pip install -q -e ".[onnx]" && python scripts/colab_run.py baselines
```

Extra arguments go to `radar_cls.experiments run`, e.g. `--seeds 0 1 --set train.epochs=40`.
Commit the generated `splits/*.csv` so every platform uses the same test images.

## Project structure

```
├── configs/
│   ├── default.yaml          # full model, clean protocol
│   ├── kaggle.yaml           # Kaggle paths, W&B on
│   └── experiments/          # ablation, baselines, smoke suites
├── src/radar_cls/
│   ├── config.py             # typed config, YAML loading, --set overrides
│   ├── data.py               # stratified splits, transforms, loaders
│   ├── models/               # model registry, RF-SpectroNet and baselines
│   ├── engine.py             # train/eval loops, EMA, MixUp
│   ├── train.py              # training CLI (checkpoints, resume, AMP, TensorBoard)
│   ├── evaluate.py           # metrics, per-class report, confusion matrix
│   ├── export.py             # TorchScript / ONNX export with metadata
│   ├── compress.py           # INT8 quantization (ONNX Runtime) and FP32 vs INT8 comparison
│   ├── profiling.py          # parameters, FLOPs, CPU latency
│   ├── tracking.py           # optional Weights & Biases logging
│   ├── experiments.py        # suite runner (variants x seeds) and result aggregation
│   └── predict.py            # inference CLI
├── scripts/                  # cloud_run.py (+ kaggle_run.py, colab_run.py wrappers)
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

- The dataset has no SNR labels, so robustness is measured with added image noise as a proxy, not versus SNR.
- No measurements on embedded hardware; any latency figures will be from a desktop CPU.

## Author

Giang Vinh Huy, Ho Chi Minh City University of Technology and Education (HCMUTE).

## License

MIT, see [LICENSE](LICENSE).
