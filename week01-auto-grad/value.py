class Value():
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._children = set(_children)
        self.grad = 0.0
        self._op = _op
    
    # string stuff
    def __repr__(self):
        return f'Value | data = {self.data}'
    
    # addition
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out
    
    # multiplication
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out

    # exponent
    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError(f"exponent must be int or float, got {type(other)}")
        out = Value(self.data ** other, (self,), f'**{other}')
        return out


# testing area
a = Value(3.0)
b = a ** 2
print(b)       # Value | data = 9.0
print(b._op)   # **2
c = a ** 'hello'
print(b) # TypeError