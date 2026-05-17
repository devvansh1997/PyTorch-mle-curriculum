from nn import MLP

# data
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
# ground truth
ys = [1.0, -1.0, -1.0, 1.0]

mlp = MLP(4, [4, 2, 1])

for epoch in range(100):

    # forward pass & compute loss
    ypred = [mlp(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # zero grad
    for p in mlp.parameters():
        p.grad = 0.0

    # backward pass
    loss.backward()

    # update weights
    for p in mlp.parameters():
        p.data -= 0.05 * p.grad
    
    print(f"Results - Epoch: {epoch} | Loss: {loss.data}")

# print preds
ypred = [mlp(x) for x in xs]
print("predictions:", [yp.data for yp in ypred])
print("targets:    ", ys)