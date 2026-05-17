import torch
from value import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).tanh()
d += 3 * d + (b - a).tanh()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
g.backward()
print(f"Value: g = {g.data:.4f}, a.grad = {a.grad:.4f}, b.grad = {b.grad:.4f}")

# Torch version
a = torch.tensor(-4.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
c = a + b
d = a * b + b**3
c = c + c + 1
c = c + 1 + c + (-a)
d = d + d * 2 + (b + a).tanh()
d = d + 3 * d + (b - a).tanh()
e = c - d
f = e**2
g = f / 2.0
g = g + 10.0 / f
g.backward()
print(f"Torch: g = {g.item():.4f}, a.grad = {a.grad.item():.4f}, b.grad = {b.grad.item():.4f}")