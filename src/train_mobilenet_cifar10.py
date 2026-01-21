import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CIFAR-10 normalization
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    ),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    ),
])

def main():
    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False)

    model = models.mobilenet_v2(weights="DEFAULT")
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0

    for epoch in range(1, 6):  # 5 epochs only
        model.train()
        pbar = tqdm(trainloader, desc=f"Epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.detach()))

        # evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch}: test accuracy = {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "results/mobilenet_v2_cifar10.pt")
            print("Saved: results/mobilenet_v2_cifar10.pt")

    print("Best test accuracy:", best_acc)

if __name__ == "__main__":
    main()
