import torch

# === block 1: data ===
# training
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

# testing
ys = [1.0, -1.0, -1.0, 1.0]

# create tensors
X = torch.tensor(xs, dtype=torch.float32)
y = torch.tensor(ys, dtype=torch.float32)

# print(X.shape)
# print(y.shape)

# === block 2: parameters ===
torch.manual_seed(0)

# input layer
W1 = torch.randn(3, 4, requires_grad=True)
b1 = torch.zeros(4, requires_grad=True)

# hidden layer 1
W2 = torch.randn(4, 2, requires_grad=True)
b2 = torch.zeros(2, requires_grad= True)

# output layer
W3 = torch.randn(2, 1, requires_grad=True)
b3 = torch.zeros(1, requires_grad=True)

# all params
params = [W1, b1, W2, b2, W3, b3]

# for p in params:
#     print(p.shape, p.requires_grad)

# block 3: the forward pass
def forward(X):
    h1 = (X @ W1 + b1).tanh()
    h2 = (h1 @ W2 + b2).tanh()
    out = (h2 @ W3 + b3).tanh()
    return out.squeeze(-1)

out = forward(X)
# print("output shape:", out.shape)
# print("output values:", out)
# print("grad_fn:", out.grad_fn)
# print("requires_grad:", out.requires_grad)
# print("is_leaf:", out.is_leaf)

# block 4: loss + backward pass
def loss_fn(out, y):
    return ((out - y)**2).sum()

loss = loss_fn(out, y)
print(f"Loss: {loss} | Shape: {loss.shape}")

loss.backward()

# for name, p in zip(["W1","b1","W2","b2","W3","b3"], params):
#     print(f"{name}: param shape {tuple(p.shape)} | grad shape {tuple(p.grad.shape)} | grad mean {p.grad.abs().mean().item():.4f}")

# block 5: the training loop
lr = 0.005
n_epochs = 100

for epoch in range(n_epochs):
    # zero gradients from previous steps
    for p in params:
        p.grad = None

    # forward pass + loss
    out = forward(X)
    loss = loss_fn(out, y)

    # backward
    loss.backward()

    # params update - must be under no_grad
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

# get predictions
with torch.no_grad():
    final_out = forward(X)
print(f"Predictions: {final_out.tolist()}")
print(f"Targets:     {y.tolist()}")