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

    def __pow__(self , other):
      assert isinstance(other, (int, float)), "only supporting int/float powers for now"
      out = Scalar(self.data**other , (self,), f'**{other}')
      def _backward():
        self.grad += (other * self.data**(other-1)) * out.grad
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

    def __radd(self , other):
      return self + other
        
    def __rsub__(self, other): 
      return other + (-self)

    def __rtruediv__(self, other):
      return other * self**-1

    def __repr__(self):
      return f"Scalar(data={self.data} , grad={self.grad})"
    


