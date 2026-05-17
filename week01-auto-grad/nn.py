from value import Value
import random

class Neuron():
    def __init__(self, nin):
        self.nin = nin
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        act = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        return act.tanh()
    
    def parameters(self):
        return self.w + [self.b]

class Layer():
    def __init__(self, nin, nout):
        self.nin = nin
        self.nout = nout
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        # edge case
        if self.nout == 1:
            return self.neurons[0](x)
        
        # output list
        out = []
        # for each neuron in our layer we process the input
        for neuron in self.neurons:
            out.append(neuron(x))
        
        return out
    
    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

class MLP():
    def __init__(self, nin, nouts):
        self.nin = nin
        self.nout = nouts
        sz = [nin] + nouts      # [3, 4, 4, 1]
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params