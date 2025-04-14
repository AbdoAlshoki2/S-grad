# S-grad: A Simple Differentiable Computation Engine

S-grad is a lightweight, educational implementation of automatic differentiation for building and training neural networks from scratch. This project was created as part of my educational journey in deep learning and NLP, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd/tree/master) and his [video tutorial](https://www.youtube.com/watch?v=VMj-3S1tku0).

## Overview

S-grad implements a scalar-valued autograd engine that powers a simple neural network library. The project consists of two main components:

1. **Engine (`engine.py`)**: A scalar-valued automatic differentiation engine that tracks computations and computes gradients.
2. **Neural Network (`nn.py`)**: A neural network library built on top of the engine, providing classes for neurons, layers, and multi-layer perceptrons.

## Features

- **Automatic Differentiation**: Track computations and compute gradients automatically
- **Neural Network Building Blocks**:
  - `Neuron`: Basic computational unit with configurable activation functions
  - `Layer`: Collection of neurons
  - `MLP`: Multi-layer perceptron with customizable architecture
- **Activation Functions**: Support for various activation functions:
  - ReLU
  - Sigmoid
  - Tanh
  - Linear (no activation)
- **Mathematical Operations**: Support for basic operations like addition, multiplication, division, and more advanced functions like sin, exp, log

## Installation

Clone the repository:

```bash
git clone https://github.com/AbdoAlshoki2/S-grad.git
cd S-grad
```
## Usage
Here's a simple example of creating and using a neural network:
```python
from s_grad import Neuron, Layer, MLP, Scalar
import numpy as np

# Create a multi-layer perceptron with:
# - 2 input features
# - 3 neurons in the first hidden layer with tanh activation
# - 4 neurons in the second hidden layer with relu activation
# - 5 neurons in the third hidden layer with sigmoid activation
# - 1 output neuron with tanh activation
model = MLP(2, [3, 4, 5, 1], ['tanh', 'relu', 'sigmoid', 'tanh'])

# Example input
x = [Scalar(0.5), Scalar(0.3)]

# Forward pass
output = model(x)

# Set a target value
target = Scalar(1.0)

# Compute loss (e.g., MSE)
loss = (output - target)**2

# Backward pass to compute gradients
loss.backward()

# Access parameters and their gradients
params = model.parameters()
for p in params:
    print(f"Parameter: {p.data}, Gradient: {p.grad}")
```

## Educational Purpose
This project is part of my educational journey in deep learning and NLP. It follows my previous educational repository where I implemented the [BPE tokenization algorithm](https://github.com/AbdoAlshoki2/Tokenizer-in-py).

The main goals of this project are:

1. Understand the fundamentals of automatic differentiation and gain practical insight into how frameworks like PyTorch implement their autograd functionality
2. Learn how neural networks work from the ground up
3. Gain insights into backpropagation and gradient-based optimization
4. Build a foundation for more advanced deep learning concepts

## Implementation Details
### Scalar Class
The Scalar class is the core of the autograd engine. It wraps a scalar value and tracks the computational graph for gradient computation. Key methods include:

- Basic arithmetic operations ( __add__ , __mul__ , etc.)
- Activation functions ( relu , sigmoid , tanh )
- Mathematical functions ( sin , exp , log )
- Backward pass for gradient computation

### Neural Network Components
The neural network library provides:

- Module : Base class with common functionality
- Neuron : Single neuron with weights, bias, and activation
- Layer : Collection of neurons with the same activation
- MLP : Multi-layer perceptron with customizable architecture