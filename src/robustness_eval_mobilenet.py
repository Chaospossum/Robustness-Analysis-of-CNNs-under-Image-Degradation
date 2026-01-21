import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, models
import numpy as np
import pandas as pd
from tqdm import tqdm

from degradations import (
    gaussian_noise, gaussian_blur, darken,
    contrast, jpeg_compress, motion_blur
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NORMALIZE = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)

def evaluate(model, loader, degrade_fn=None):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, leave=False):
            if degrade_fn is not None:
                x_np = x.permute(0, 2, 3, 1).numpy()
                x_np = np.stack([degrade_fn(img) for img in x_np], axis=0)
                x = torch.from_numpy(x_np).permute(0, 3, 1, 2)

            x = NORMALIZE(x)
            x, y = x.to(DEVICE), y.to(DEVICE)

            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total

def main():
    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(testset, batch_size=128, shuffle=False)

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    model.load_state_dict(torch.load("results/mobilenet_v2_cifar10.pt", map_location=DEVICE))
    model = model.to(DEVICE)

    results = []

    # Baseline
    acc = evaluate(model, loader)
    results.append(("baseline", 0.0, acc))
    print("Baseline accuracy:", acc)

    # Noise
    for sigma in [0.02, 0.05, 0.1]:
        acc = evaluate(model, loader, lambda img: gaussian_noise(img, sigma))
        results.append(("gaussian_noise", float(sigma), acc))
        print(f"Noise sigma={sigma}: {acc}")

    # Blur
    for k in [3, 5, 7]:
        acc = evaluate(model, loader, lambda img: gaussian_blur(img, k))
        results.append(("gaussian_blur", float(k), acc))
        print(f"Blur k={k}: {acc}")

    # Darken
    for f in [0.8, 0.6, 0.4]:
        acc = evaluate(model, loader, lambda img: darken(img, f))
        results.append(("darken", float(f), acc))
        print(f"Darken factor={f}: {acc}")

    # Contrast
    for c in [0.8, 0.6, 0.4]:
        acc = evaluate(model, loader, lambda img: contrast(img, c))
        results.append(("contrast", float(c), acc))
        print(f"Contrast factor={c}: {acc}")

    # JPEG (note: lower quality = worse)
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
    df.to_csv("results/robustness_mobilenet_v2.csv", index=False)
    print("Saved: results/robustness_mobilenet_v2.csv")

if __name__ == "__main__":
    main()
