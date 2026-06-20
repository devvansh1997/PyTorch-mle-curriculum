import torch
import torch.nn as nn


class BasicCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # size (B, 16, 16, 16)
        x = self.pool(self.relu(self.conv2(x)))  # szie (B, 32, 8, 8)
        x = torch.flatten(x, start_dim=1)  # size (B, 32 * 8 * 8)
        return self.fc(x)  # returns the raw logits for CrossEntropyLoss fx
