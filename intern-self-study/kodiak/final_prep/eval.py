import torch
import torch.nn as nn


# disables gradient calc
@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    # set model to eval mode
    model.eval()

    correct = total = 0
    # eval loop
    for x, y in loader:
        # move data to device
        x, y = x.to(device), y.to(device)

        # get predictions
        preds = model(x).argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total
