import pandas as pd

r = pd.read_csv("results/robustness_resnet18_cifartrained.csv")
m = pd.read_csv("results/robustness_mobilenet_v2.csv")

r["model"] = "ResNet-18"
m["model"] = "MobileNet-V2"

df = pd.concat([r, m], ignore_index=True)
df = df[["model", "degradation", "level", "accuracy"]]
df.to_csv("results/robustness_both_models.csv", index=False)

print("Saved: results/robustness_both_models.csv")
print(df.head(15).to_string(index=False))
