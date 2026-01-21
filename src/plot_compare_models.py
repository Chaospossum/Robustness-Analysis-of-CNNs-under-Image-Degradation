import pandas as pd
import matplotlib.pyplot as plt

resnet = pd.read_csv("results/robustness_resnet18_cifartrained.csv")
mobilenet = pd.read_csv("results/robustness_mobilenet_v2.csv")

degradations = sorted(set(resnet["degradation"]) & set(mobilenet["degradation"]))

for degr in degradations:
    if degr == "baseline":
        continue

    r = resnet[resnet["degradation"] == degr].sort_values("level")
    m = mobilenet[mobilenet["degradation"] == degr].sort_values("level")

    plt.figure(figsize=(7,4))
    plt.plot(r["level"], r["accuracy"], marker="o", label="ResNet-18")
    plt.plot(m["level"], m["accuracy"], marker="o", label="MobileNet-V2")

    plt.xlabel("Degradation level")
    plt.ylabel("Accuracy")
    plt.title(f"Robustness comparison: {degr}")
    plt.grid(True)
    plt.legend()

    out = f"figures/compare_{degr}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("Saved:", out)
