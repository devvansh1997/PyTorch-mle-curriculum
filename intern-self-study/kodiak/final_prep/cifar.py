import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class CIFAR_Model(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool(self.relu(self.batchnorm2(self.conv2(x))))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        return self.fc(x)


def train(model, train_loader, val_loader, epochs=10, lr=1e-3, device="cpu"):
    model.to(device)

    loss_fx = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fx(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch: {epoch + 1} | Loss: {total_loss / len(train_loader):.4f}")
        # Validation
        print(f"Validation Accuracy: {evaluate(model, val_loader, device)}")


@torch.no_grad()
def evaluate(model, test_dataloader, device="cpu"):
    model.to(device)

    model.eval()
    total = correct = 0
    for x, y in test_dataloader:
        x, y = x.to(device), y.to(device)
        preds = torch.argmax(model(x), dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


if __name__ == "__main__":
    # setup transforms before importing dataset
    cifar_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # import the data here for now
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=cifar_transforms
    )

    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=cifar_transforms
    )

    # partition train and val sizes
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size

    # split train dataset into train, val
    train_sub, val_sub = random_split(train_dataset, [train_size, val_size])

    # dataloaders
    train_ds = DataLoader(train_sub, batch_size=32, shuffle=True)
    val_ds = DataLoader(val_sub, batch_size=32, shuffle=False)
    test_ds = DataLoader(test_dataset, batch_size=32, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CIFAR_Model()
    # begin training
    train(model, train_ds, val_ds, device=device)
    # begin evaluation
    accuracy = evaluate(model, test_ds, device)
    print(f"Final Test Accuracy: {accuracy:.3f}")
