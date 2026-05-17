import math

class Value():
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._children = set(_children)
        self.grad = 0.0
        self._op = _op
        self._backward = lambda: None
    
    # string stuff
    def __repr__(self):
        return f'Value | data = {self.data}'
    
    # addition
    def __add__(self, other):
        # check for a regular number
        if isinstance(other, int) or isinstance(other, float):
            other = Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    # multiplication
    def __mul__(self, other):
        # check for a regular number
        if isinstance(other, int) or isinstance(other, float):
            other = Value(other)

        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    # exponent
    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError(f"exponent must be int or float, got {type(other)}")
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out
    
    # negative
    def __neg__(self):
        out = self.__mul__(Value(-1))
        return out
    
    # substract ( for gradients )
    def __sub__(self, other):
        out = self.__add__(other.__neg__())
        return out
    
    # divide ( for normalization )
    def __truediv__(self, other):
        return self * other ** -1
    
    # reverse multiply
    def __rmul__(self, other):
        return self * other
    
    # reverse addition
    def __radd__(self, other):
        return self + other
    
    # reverse substraction
    def __rsub__(self, other):
        return other + (-self)
    
    # reverse divide
    def __rtruediv__(self, other):
        return other * self ** -1
    
    def backward(self):
        # topological sort
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # seed the gradient
        self.grad = 1.0

        # walk backwards on the graph
        for node in reversed(topo):
            node._backward()
    
    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
# testing area
# a = Value(4.0)
# b = Value(2.0)
# c = a / b
# c.backward()
# print(c.data)   # 2.0
# print(a.grad)   # 0.5    (∂(a/b)/∂a = 1/b)
# print(b.grad)   # -1.0   (∂(a/b)/∂b = -a/b² = -4/4)

# print(8 / a)    # Value | data = 2.0