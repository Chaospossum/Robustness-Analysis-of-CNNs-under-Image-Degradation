# Robustness Analysis of CNNs under Image Degradation

This repository contains the code and results for a bachelor thesis on the
robustness of convolutional neural networks (CNNs) to image degradations.

## Models

* ResNet-18 (trained on CIFAR-10)
* MobileNet-V2 (fine-tuned on CIFAR-10)

## Degradations

* Gaussian noise
* Gaussian blur
* Motion blur
* JPEG compression
* Contrast reduction
* Darkening

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python3 src/train_cifar10_resnet18.py
python3 src/train_mobilenet_cifar10.py
```

## Robustness evaluation

```bash
python3 src/robustness_eval.py
python3 src/robustness_eval_mobilenet.py
```

## Analysis and plots

```bash
python3 src/threshold_analysis.py
python3 src/plot_compare_models.py
```

## Notes

Model weights are not included in the repository.
All results can be reproduced by running the provided training and evaluation scripts.
Robustness analysis is based on accuracy degradation and a 50% accuracy failure threshold.
