import torch
import torch.nn as nn
from pandas.compat.pickle_compat import load


def train(model, loader, epochs=2, lr=1e-3, device="cpu"):
    # move model to device: cpu or gpu
    model.to(device)

    # establish the loss fx and optimizer
    loss_fx = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # set model to training mode
    model.train()

    # start training
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in loader:
            # move data to device
            x, y = x.to(device), y.to(device)

            # clear gradients
            optimizer.zero_grad()
            # raw logits from model output
            logits = model(x)
            # calc loss
            loss = loss_fx(logits, y)
            # calc gradients
            loss.backward()
            # step optimizer
            optimizer.step()

            # accumulate loss
            total_loss += loss.item()
        # show training loop with loss
        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(loader):.4f}")
