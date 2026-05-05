# RF Signal Spectrum Classification using Lightweight CNN

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-91.83%25-brightgreen.svg)](#performance-results)

A high-performance deep learning system for automatic identification of **Radio Frequency (RF)** signals from spectrograms. This project achieves state-of-the-art accuracy within a strict hardware budget of **fewer than 100,000 parameters**, optimized for **Edge AI** and **TinyML** applications.

---

## Key Highlights

- **Optimized Architecture:** Custom **BasicCNN** (RF-SpectroNet) using **Mobile Inverted Bottleneck (MBConv)** blocks with integrated **Coordinate Attention (CoordAtt)** for spatial-temporal feature capture.
- **High Efficiency:** Achieved **91.83% validation accuracy** with only **89,196 parameters**.
- **Robust Training:** MixUp augmentation, Label Smoothing, Exponential Moving Average (EMA), and Cosine Annealing schedule.
- **Deployment Ready:** Models exported via **TorchScript** for seamless integration into C++ or embedded real-time monitoring systems.

---

## Dataset & Classification

The model classifies **76,800 spectrogram images** (resized to 224x224) into 12 specialized classes:

| Category | Signal Classes |
|:---|:---|
| **Telecommunications** | 16-QAM, B-FM, BPSK, CPFSK, DSB-AM, GFSK, PAM4, QPSK |
| **Radar** | Barker, LFM, Rect, Step-FM |

---

## Project Structure

```
project_cuoiki/
├── src/
│   └── Main.py            # Training & evaluation script
├── model/
│   └── TrainedModel.pt     # TorchScript exported model
├── docs/
│   └── Report.pdf          # Full technical report
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/vinhhuy15/Radar-Signal-Classification-DL.git
cd Radar-Signal-Classification-DL

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python src/Main.py
```

> **Note:** The training script expects the dataset at a Kaggle-style path. Modify the `data_dir` variable in `src/Main.py` to point to your local dataset directory.

### Inference

```python
import torch

# Load the TorchScript model
model = torch.jit.load("model/TrainedModel.pt")
model.eval()

# Run inference on a spectrogram image
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)
predicted_class = torch.argmax(output, dim=1)
```

---

## Technical Details

| Component | Choice |
|:---|:---|
| **Architecture** | BasicCNN with MBConv + Coordinate Attention |
| **Pooling** | Hybrid (Global Average + Max Pooling concatenation) |
| **Regularization** | MixUp (alpha=0.2), Label Smoothing (0.1), Dropout (0.3) |
| **Optimizer** | AdamW (lr=2e-3, weight_decay=1e-2) |
| **Scheduler** | Cosine Annealing (T_max=80, eta_min=1e-6) |
| **Weight Averaging** | EMA (decay=0.999) |

---

## Performance Results

| Model | Parameters | Accuracy | F1-Score |
|:---|:---|:---|:---|
| Baseline | 88,524 | 88.30% | 89.48% |
| **BasicCNN (Ours)** | **89,196** | **91.83%** | **90.35%** |

---

## Authors

**Ho Chi Minh City University of Technology and Education (HCMUTE)**

- Giang Vinh Huy

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{vhhuy2026rfclassification,
  title     = {RF Signal Spectrum Classification using Lightweight CNN},
  author    = {Giang Vinh Huy},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/vinhhuy15/Radar-Signal-Classification-DL}
}
```
