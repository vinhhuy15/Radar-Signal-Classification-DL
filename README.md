# RF Signal Spectrum Classification using Lightweight CNN

A high-performance deep learning system designed for the automatic identification of **Radio Frequency (RF)** signals from spectrograms. [cite_start]This project achieves state-of-the-art accuracy within a strict hardware budget of **fewer than 100,000 parameters**, optimized for **Edge AI** and **TinyML** applications[cite: 89, 416, 1101].

## Key Highlights
* [cite_start]**Optimized Architecture:** Custom **BasicCNN** (RF-SpectroNet) using **Mobile Inverted Bottleneck (MBConv)** blocks[cite: 416, 422].
* [cite_start]**Spatial-Temporal Awareness:** Integrated **Coordinate Attention (CoordAtt)** to capture signal trajectories in noisy environments[cite: 493, 502].
* [cite_start]**High Efficiency:** Achieved **91.83% validation accuracy** with only **89,196 parameters**[cite: 63, 67].
* [cite_start]**Deployment Ready:** Models are exported via **TorchScript** for seamless integration into C++ or embedded real-time monitoring systems[cite: 601, 632].

---

## Dataset & Classification
[cite_start]The model classifies **76,800 spectrogram images** (resized to $224 \times 224$) into 12 specialized classes[cite: 243, 247, 372]:

| Category | Signal Classes |
| :--- | :--- |
| **Telecommunications** | [cite_start]16-QAM, B-FM, BPSK, CPFSK, DSB-AM, GFSK, PAM4, QPSK [cite: 251] |
| **Radar** | [cite_start]Barker, LFM, Rect, Step-FM [cite: 263] |

---

## Technical Implementation
[cite_start]To ensure robustness against environmental noise (Low SNR), the following R&D techniques were applied[cite: 84, 185, 365]:

1. [cite_start]**Hybrid Feature Aggregation:** Concatenates **Global Average** and **Max Pooling** to detect both stable telecom patterns and sharp radar pulses[cite: 535, 540].
2. [cite_start]**Robust Training:** Utilized **MixUp Augmentation** and **Label Smoothing (0.1)** to prevent overfitting and improve generalization[cite: 312, 332, 577].
3. [cite_start]**Weight Smoothing:** Implemented **Exponential Moving Average (EMA)** with a decay of 0.999 to stabilize convergence[cite: 333, 567].
4. [cite_start]**Optimizer:** **AdamW** paired with a **Cosine Annealing** learning rate scheduler for optimal fine-tuning[cite: 588, 592].

---

## 📈 Performance Results
[cite_start]The proposed architecture provides a significant leap in efficiency over standard baselines[cite: 1085, 1097]:

| Model | Parameters | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- |
| Baseline | 88,524 | 88.30% | [cite_start]89.48% [cite: 67] |
| **BasicCNN (Ours)** | **89,196** | **91.83%** | [cite_start]**90.35%** [cite: 67] |

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
