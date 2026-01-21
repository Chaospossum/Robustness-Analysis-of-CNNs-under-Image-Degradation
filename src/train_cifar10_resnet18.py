import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, models
from tqdm import tqdm
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CIFAR-10 normalization
MEAN = [0.4914, 0.4822, 0.4465]
STD  = [0.2023, 0.1994, 0.2010]

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def main():
    os.makedirs("results", exist_ok=True)

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    testset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    test_loader  = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    # ResNet-18 adapted for CIFAR-10 (32x32)
    model = models.resnet18(weights=None, num_classes=10)
    # Better for CIFAR: remove the first maxpool and use smaller conv
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

    best = 0.0
    EPOCHS = 5  # keep it fast; you can increase later

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss))

        scheduler.step()

        acc = accuracy(model, test_loader)
        print(f"Epoch {epoch}: test accuracy = {acc:.4f}")

        if acc > best:
            best = acc
            torch.save(model.state_dict(), "results/resnet18_cifar10.pt")
            print("Saved: results/resnet18_cifar10.pt")

    print("Best test accuracy:", best)

if __name__ == "__main__":
    main()

