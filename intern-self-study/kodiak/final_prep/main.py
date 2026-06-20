# import matplotlib.pyplot as plt
import torch
from data import SyntheticDataset
from eval import evaluate
from model import BasicCNN
from torch.utils.data import DataLoader
from train import train

# create a dataset
my_syn_dataset_train = SyntheticDataset(n=1000, num_classes=10)
my_syn_dataset_test = SyntheticDataset(n=200, num_classes=10)

# create the dataloaders
train_ds = DataLoader(my_syn_dataset_train, batch_size=32, shuffle=True)
test_ds = DataLoader(my_syn_dataset_test, batch_size=32, shuffle=True)

# execute training loop

# setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# instantiate the model
model = BasicCNN()

# start training
train(model, train_ds, epochs=100, device=device)

# perform evaluation
accuracy = evaluate(model, test_ds, device=device)
print(f"Final Accuracy for classification model: {accuracy:.3f}")
