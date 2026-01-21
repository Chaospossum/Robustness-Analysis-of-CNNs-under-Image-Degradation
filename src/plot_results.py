import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/robustness_resnet18_cifartrained.csv")

plt.figure(figsize=(8,5))

for degr in df["degradation"].unique():
    sub = df[df["degradation"] == degr].sort_values("level")
    plt.plot(sub["level"], sub["accuracy"], marker="o", label=degr)

plt.xlabel("Degradation level")
plt.ylabel("Accuracy")
plt.title("Robustness of CIFAR-10 trained ResNet-18")
plt.grid(True)
plt.legend()

plt.savefig("figures/robustness_resnet18_cifartrained.png", dpi=300, bbox_inches="tight")
print("Saved: figures/robustness_resnet18_cifartrained.png")
