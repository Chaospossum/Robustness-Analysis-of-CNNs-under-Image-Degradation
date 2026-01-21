import pandas as pd

THRESHOLD = 0.5

files = {
    "ResNet-18": "results/robustness_resnet18_cifartrained.csv",
    "MobileNet-V2": "results/robustness_mobilenet_v2.csv",
}

rows = []

for model, path in files.items():
    df = pd.read_csv(path)

    for degr in df["degradation"].unique():
        if degr == "baseline":
            continue

        sub = df[df["degradation"] == degr].sort_values("level")

        failed = sub[sub["accuracy"] < THRESHOLD]

        if len(failed) == 0:
            level = "Not reached"
        else:
            level = failed.iloc[0]["level"]

        rows.append({
            "model": model,
            "degradation": degr,
            "failure_level (acc < 0.5)": level
        })

out = pd.DataFrame(rows)
out.to_csv("results/robustness_thresholds.csv", index=False)

print(out.to_string(index=False))
print("Saved: results/robustness_thresholds.csv")
