# Custom Autograd Engine

This is a simple, from-scratch autograd engine that performs automatic differentiation for scalar values. It supports basic operations like addition, multiplication, exponentiation, and ReLU activation. The engine dynamically builds a computation graph and computes gradients using backpropagation.

## Overview

The `Value` class tracks scalar values and their gradients during forward and backward passes. It supports simple arithmetic operations and computes the gradient through the chain rule during backpropagation. This implementation is designed to help you understand the core concepts behind automatic differentiation and backpropagation.

## Features

- Basic operations: `+`, `-`, `*`, `/`, `**` (Exponentiation)
- ReLU activation function
- Automatic gradient calculation with the `backward()` method
- Intuitive API for performing computations

## Example Usage

Below is an example of how to use the custom autograd engine to perform a forward pass and compute gradients using backpropagation.

### Code Example

```python
from micrograd.engine import Value

# Define scalar values
a = Value(-4.0)
b = Value(2.0)

# Perform some operations
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()

# Compute the final result
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f

# Print the final result after the forward pass
print(f'{g.data:.4f}')  # prints 24.7041, the outcome of this forward pass

# Backpropagation to compute gradients
g.backward()

# Print the gradients of 'a' and 'b'
print(f'{a.grad:.4f}')  # prints 138.8338, the gradient of g with respect to a
print(f'{b.grad:.4f}')  # prints 645.5773, the gradient of g with respect to b
