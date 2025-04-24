import math
from autograd.tensor import Value

def relu(x):
    out = Value(x.data if x.data > 0 else 0, (x,), 'ReLU')

    def _backward():
        x.grad += (x.data > 0) * out.grad
    out._backward = _backward
    return out

def sigmoid(x):
    s = 1 / (1 + math.exp(-x.data))
    out = Value(s, (x,), 'sigmoid')

    def _backward():
        x.grad += s * (1 - s) * out.grad
    out._backward = _backward
    return out

def mse_loss(preds, targets):
    losses = [(p - t) ** 2 for p, t in zip(preds, targets)]
    return sum(losses) * (1 / len(losses))
