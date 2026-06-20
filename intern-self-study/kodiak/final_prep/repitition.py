import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class SyntheticDataset(Dataset):
    def __init__(self, n=1000):
        self.images = torch.randn(n, 3, 32, 32)
        self.labels = torch.randn(n, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class EasyCNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool(self.relu(self.batchnorm2(self.conv2(x))))
        x = torch.flatten(x, start_dim=1)
        return self.fc(self.dropout(x))


def train(model, train_dataloader, val_dataloader, epochs=10, lr=1e-2, device="cpu"):
    model.to(device)

    loss_fx = nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
    )

    for epoch in range(epochs):
        # training
        model.train()
        total_loss = 0.0
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fx(model(x).squeeze(-1), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch: {epoch + 1} | Loss: {total_loss / len(train_dataloader):.4f}")

        # validation
        print(f"Validation MSE: {evaluate(model, val_dataloader):.4f}")


@torch.no_grad()
def evaluate(model, dataloader, device="cpu"):
    model.to(device)
    loss = 0.0
    model.eval()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss += nn.functional.mse_loss(pred.squeeze(-1), y)
    return loss
