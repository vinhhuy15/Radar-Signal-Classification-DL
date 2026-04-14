# RF Signal Spectrum Classification using Lightweight CNN

A high-performance deep learning system designed for the automatic identification of **Radio Frequency (RF)** signals from spectrograms. This project achieves state-of-the-art accuracy within a strict hardware budget of **fewer than 100,000 parameters**, optimized for **Edge AI** and **TinyML** applications.

## Key Highlights
* **Optimized Architecture:** Custom **BasicCNN** (RF-SpectroNet) using **Mobile Inverted Bottleneck (MBConv)** blocks.
* **Spatial-Temporal Awareness:** Integrated **Coordinate Attention (CoordAtt)** to capture signal trajectories in noisy environments.
* **High Efficiency:** Achieved **91.83% validation accuracy** with only **89,196 parameters**.
* **Deployment Ready:** Models are exported via **TorchScript** for seamless integration into C++ or embedded real-time monitoring systems.

---

## Dataset & Classification
The model classifies **76,800 spectrogram images** (resized to $224 \times 224$) into 12 specialized classes:

| Category | Signal Classes |
| :--- | :--- |
| **Telecommunications** | 16-QAM, B-FM, BPSK, CPFSK, DSB-AM, GFSK, PAM4, QPSK |
| **Radar** | Barker, LFM, Rect, Step-FM |

---

## Technical Implementation
To ensure robustness against environmental noise (Low SNR), the following R&D techniques were applied:

1. **Hybrid Feature Aggregation:** Concatenates **Global Average** and **Max Pooling** to detect both stable telecom patterns and sharp radar pulses.
2. **Robust Training:** Utilized **MixUp Augmentation** and **Label Smoothing (0.1)** to prevent overfitting and improve generalization.
3. **Weight Smoothing:** Implemented **Exponential Moving Average (EMA)** with a decay of 0.999 to stabilize convergence.
4. **Optimizer:** **AdamW** paired with a **Cosine Annealing** learning rate scheduler for optimal fine-tuning.

---

## 📈 Performance Results
The proposed architecture provides a significant leap in efficiency over standard baselines:

| Model | Parameters | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- |
| Baseline | 88,524 | 88.30% | 89.48% |
| **BasicCNN (Ours)** | **89,196** | **91.83%** | **90.35%** |

---

## 🛠 Usage
### Requirements
* Python 3.12+
* PyTorch 2.1+
* Torchvision

### Quick Start
```python
import torch

# Load the TorchScript model for inference
model = torch.jit.load("09_DeepLearningProject_TrainedModel.pt")
model.eval()

# Example inference
# input_tensor = torch.randn(1, 3, 224, 224)
# output = model(input_tensor)
