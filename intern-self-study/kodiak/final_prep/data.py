import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(self, n=1000, num_classes=10):
        # generate it myself using randn
        self.images = torch.randn(n, 3, 32, 32)  # (N, C, H, W)
        self.labels = torch.randint(0, num_classes, (n,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
