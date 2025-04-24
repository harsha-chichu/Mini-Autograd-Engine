# Mini Autograd Engine

A minimal autograd engine implemented from scratch in Python, inspired by PyTorch's computational graph and backward pass.

## Features
- Supports basic operations: addition, multiplication, tanh
- Backpropagation via topological sorting
- Gradient tracking and visualization-ready nodes

## Usage

```python
from autograd.tensor import Value

a = Value(2.0)
b = Value(3.0)
c = a * b + b
c.backward()

print(c)
print(a.grad, b.grad)
