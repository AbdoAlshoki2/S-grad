import random
from micrograd.engine import Value
from typing import Union
class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin: Union['rely' , 'sigmoid' , 'tanh'] = 'linear'):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        if self.nonlin == 'relu':
            return act.relu()
        elif self.nonlin == 'sigmoid':
            return act.sigmoid()
        elif self.nonlin =='tanh':
            return act.tanh()
        else:
            return act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.nonlin} Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts, nonlins=None):

        if nonlins is None:
            nonlins = ['relu'] * (len(nouts) - 1) + ['linear']
        elif len(nonlins) != len(nouts):
            raise ValueError(f"Length of nonlins ({len(nonlins)}) must match length of nouts ({len(nouts)})")
        
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=nonlins[i]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"