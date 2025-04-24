import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from autograd.nn import MLP
from autograd.tensor import Value

import torch

# Simple tensor initialization
x = torch.tensor([2.0, -3.0], requires_grad=True)  # A tensor with requires_grad=True so we can calculate gradients

# Performing operations on the tensor
y = x * 2 + 3  # Simple linear operation: y = 2 * x + 3
z = y.relu()    # Apply ReLU activation: ReLU(x) = max(0, x)

# Compute gradients
z.sum().backward()  # We compute the gradients of the sum of z with respect to x

# Print results
print("Tensor x:", x)
print("Tensor y:", y)
print("Tensor z (after ReLU):", z)

# Print gradients
print("Gradients of x:", x.grad)
