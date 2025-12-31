import torch
import math

# Use GPU if available, otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a tensor with random data
x = torch.randn(5, 3, device=device)
y = torch.randn(5, 3, device=device)

print(f"Tensor X:\n{x}")
print(f"Tensor Y:\n{y}")

# Perform a simple addition operation
z = x + y
print(f"Sum of X and Y:\n{z}")

# Another common operation: matrix multiplication
# Ensure dimensions are compatible for matmul if needed.
# Example with manual forward/backward pass (from basic tutorials):
# Create random input and output data for a simple fit
x_fit = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=torch.float)
y_fit = torch.sin(x_fit)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=torch.float, requires_grad=True)
b = torch.randn((), device=device, dtype=torch.float, requires_grad=True)
c = torch.randn((), device=device, dtype=torch.float, requires_grad=True)
d = torch.randn((), device=device, dtype=torch.float, requires_grad=True)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x_fit + c * x_fit ** 2 + d * x_fit ** 3

    # Compute and print loss
    loss = (y_pred - y_fit).pow(2).sum()
    if t % 500 == 0:
        print(f"Epoch {t}: Loss = {loss.item()}")

    # Use autograd to compute the backward pass (gradients)
    loss.backward()

    # Manually update weights using gradient descent.
    # Wrap in torch.no_grad() because we don't want to track these operations
    # in the autograd graph.
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # Zero the gradients after updating
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'\nResult: y = {a.item()} + {b.item()}x + {c.item()}x^2 + {d.item()}x^3')
