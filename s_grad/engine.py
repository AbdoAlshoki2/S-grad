import math

class Scalar:
    def __init__(self, data, _children = ()):
        self.data = data
        self._prev = set(_children)

        self._backward = lambda: None
        self.grad = 0.0


    def __add__(self,other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data , (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data , (self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data**other.data, (self, other))
        if self.data <= 0: # not implemented for negative bases
          raise ValueError("Cannot compute gradients for negative bases raised to variable powers")

        def _backward():
            self.grad += other.data * (self.data**(other.data-1)) * out.grad # power rule (n) * p ** n-1
            other.grad += (self.data**other.data) * math.log(self.data) * out.grad # ln(x) * x ** n
        
        out._backward = _backward
        return out
    

    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __rmul__(self, other):
        return self * other

    def __radd__(self , other):
        return self + other
        
    def __rsub__(self, other): 
        return other + (-self)

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        return f"Scalar(data={self.data} , grad={self.grad})"

    def sin(self):
        out = Scalar(math.sin(self.data) , (self,))
        def _backward():
            self.grad += math.cos(self.data) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Scalar(math.exp(self.data) , (self,))
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        if self.data <= 0: # not implemented for negative bases
          raise ValueError("Cannot compute log of non-positive value")
        
        out = Scalar(math.log(self.data) , (self,))
        def _backward():
            self.grad += (1.0/self.data) * out.grad 
        out._backward = _backward
        return out     

    def relu(self):
        out = Scalar(max(self.data , 0) , (self,))
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):

        neg_self = self * -1
        exp_term = neg_self.exp()
        one = Scalar(1.0)
        denom = one + exp_term
        out = one / denom
        def _backward():
            self.grad += (out.data * (1 - out.data)) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):

        pos_exp = (self * 2).exp()
        neg_exp = (self * -2).exp()
        num = pos_exp - neg_exp
        denom = pos_exp + neg_exp
        out = num / denom 
        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        return out

