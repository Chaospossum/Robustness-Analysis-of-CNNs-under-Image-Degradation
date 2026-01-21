import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, models
import numpy as np
import pandas as pd
from tqdm import tqdm

from degradations import gaussian_noise, gaussian_blur, darken, contrast, jpeg_compress, motion_blur


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CIFAR-10 normalization
NORMALIZE = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)

def evaluate(model, loader, degrade_fn=None):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in tqdm(loader, leave=False):
            # x is [B,3,32,32] float in [0,1]
            if degrade_fn is not None:
                x_np = x.permute(0, 2, 3, 1).numpy()   # [B,H,W,C]
                x_np = np.stack([degrade_fn(img) for img in x_np], axis=0)
                x = torch.from_numpy(x_np).permute(0, 3, 1, 2)  # back to [B,3,H,W]

            x = NORMALIZE(x)
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total

def build_cifar_resnet18():
    # ResNet18 adapted for CIFAR-10 (32x32)
    model = models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

def main():
    # Data (no normalization here; we normalize in evaluate())
    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(testset, batch_size=128, shuffle=False)

    # Load your trained CIFAR-10 model
    model = build_cifar_resnet18().to(DEVICE)
    model.load_state_dict(torch.load("results/resnet18_cifar10.pt", map_location=DEVICE))
    model.eval()

    results = []

    # Baseline
    acc = evaluate(model, loader, degrade_fn=None)
    results.append(("baseline", 0.0, acc))
    print("Baseline accuracy:", acc)

    # Gaussian noise levels (sigma in [0,1] scale)
    for sigma in [0.02, 0.05, 0.1]:
        acc = evaluate(model, loader, lambda img: gaussian_noise(img, sigma))
        results.append(("gaussian_noise", float(sigma), acc))
        print(f"Noise sigma={sigma}: {acc}")

    # Blur kernel sizes
    for k in [3, 5, 7]:
        acc = evaluate(model, loader, lambda img: gaussian_blur(img, k))
        results.append(("gaussian_blur", float(k), acc))
        print(f"Blur k={k}: {acc}")

    # Darkening factors
    for f in [0.8, 0.6, 0.4]:
        acc = evaluate(model, loader, lambda img: darken(img, f))
        results.append(("darken", float(f), acc))
        print(f"Darken factor={f}: {acc}")

    # Contrast reduction
    for c in [0.8, 0.6, 0.4]:
        acc = evaluate(model, loader, lambda img: contrast(img, c))
        results.append(("contrast", float(c), acc))
        print(f"Contrast factor={c}: {acc}")

    # JPEG compression (lower quality = worse)
    for q in [50, 30, 10]:
        acc = evaluate(model, loader, lambda img: jpeg_compress(img, q))
        results.append(("jpeg", float(q), acc))
        print(f"JPEG quality={q}: {acc}")

    # Motion blur
    for k in [3, 5, 7]:
        acc = evaluate(model, loader, lambda img: motion_blur(img, k))
        results.append(("motion_blur", float(k), acc))
        print(f"Motion blur k={k}: {acc}")

    df = pd.DataFrame(results, columns=["degradation", "level", "accuracy"])
    df.to_csv("results/robustness_resnet18_cifartrained.csv", index=False)
    print("Saved: results/robustness_resnet18_cifartrained.csv")

if __name__ == "__main__":
    main()
