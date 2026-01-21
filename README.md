# Robustness Analysis of CNNs under Image Degradation

This repository contains the code and results for a bachelor thesis on
the robustness of convolutional neural networks to image degradations.

## Models
- ResNet-18 (trained on CIFAR-10)
- MobileNet-V2 (fine-tuned on CIFAR-10)

## Degradations
- Gaussian noise
- Gaussian blur
- Motion blur
- JPEG compression
- Contrast reduction
- Darkening

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Training

python3 src/train_cifar10_resnet18.py
python3 src/train_mobilenet_cifar10.py

Robustness evaluation

python3 src/robustness_eval.py
python3 src/robustness_eval_mobilenet.py

Analysis & plots

python3 src/threshold_analysis.py
python3 src/plot_compare_models.py

Model weights are not included but can be reproduced by running the training scripts.


Save & exit.

---

## 5️⃣ Initialize Git and commit
From the project root:

```bash
git init
git add .
git commit -m "Final thesis code: robustness analysis of CNNs"
# cv-robustness-thesis
