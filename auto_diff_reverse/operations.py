import numpy as np
from .comp_node import CompNode

class Operation(CompNode):
    pass

class Negate(Operation):
    def __init__(self, A: CompNode):
        self.A = A
    
    def forward(self):
        self.A.forward()
        self.tensor = - self.A.tensor
    
    def backward(self, seed: np.ndarray | float):
        # f = -A
        # df/dA = -dA/dA
        self.A.backward(-seed)

class Add(Operation):
    def __init__(self, A: CompNode, B: CompNode):
        self.A = A
        self.B = B
    
    def forward(self):
        self.A.forward()
        self.B.forward()
        self.tensor = self.A.tensor + self.B.tensor
    
    def backward(self, seed: np.ndarray | float):
        # f = A + B
        # df/dA = 1 * dA/dA
        # df/dB = 1 * dB/dB
        self.A.backward(seed)
        self.B.backward(seed)

class Subtract(Operation):
    def __init__(self, A: CompNode, B: CompNode):
        self.A = A
        self.B = B
    
    def forward(self):
        self.A.forward()
        self.B.forward()
        self.tensor = self.A.tensor - self.B.tensor
    
    def backward(self, seed: np.ndarray | float):
        # f = A - B
        # df/dA = 1 * dA/dA
        # df/dB = -1 * dB/dB
        self.A.backward(seed)
        self.B.backward(-seed)

class Multiply(Operation):
    def __init__(self, A: CompNode, B: CompNode):
        self.A = A
        self.B = B
    
    def forward(self):
        self.A.forward()
        self.B.forward()
        self.tensor = self.A.tensor * self.B.tensor
    
    def backward(self, seed: np.ndarray | float):
        # f = AB
        # df/dA = B * dA/dA
        # df/dB = A * dB/dB
        self.A.backward(self.B.tensor * seed)
        self.B.backward(self.A.tensor * seed)

class Divide(Operation):
    def __init__(self, A: CompNode, B: CompNode):
        self.A = A
        self.B = B
    
    def forward(self):
        self.A.forward()
        self.B.forward()
        self.tensor = self.A.tensor / self.B.tensor
    
    def backward(self, seed: np.ndarray | float):
        # f = A/B
        # df/dA = 1/B * dA/dA
        # df/dB = -A/(B^2) * dB/dB
        self.A.backward(seed / self.B.tensor)
        self.B.backward(- self.A.tensor * seed / (self.B.tensor ** 2))

# Exponential
class Exp(Operation):

    def __init__(self, A: CompNode):
        self.A = A

    def forward(self):
        self.A.forward()
        self.tensor = np.exp(self.A.tensor)

    def backward(self, seed: np.ndarray | float):
        # f = e^A
        # df/dA = e^A * dA/dA
        self.A.backward(self.tensor * seed)

# Natural Log
class Log(Operation):

    def __init__(self, A: CompNode):
        self.A = A

    def forward(self):
        self.A.forward()
        self.tensor = np.log(self.A.tensor)

    def backward(self, seed: np.ndarray | float):
        # f = ln(A)
        # df/dA = 1/A * dA/dA
        self.A.backward(seed / self.A.tensor)

class Power(Operation):
    def __init__(self, A: CompNode, B: CompNode):
        self.A = A
        self.B = B
    
    def forward(self):
        self.A.forward()
        self.B.forward()
        self.tensor = self.A.tensor ** self.B.tensor
    
    def backward(self, seed: np.ndarray | float):
        # f = A^B
        # df/dA = B * A^(B-1) * dA/dA
        # df/dB = ln(A) * A^B * dB/dB
        self.A.backward(self.B.tensor * (self.A.tensor ** (self.B.tensor - 1.0)) * seed)
        self.B.backward(np.log(self.A.tensor) * self.tensor * seed)

class Sqrt(Operation):

    def __init__(self, A: CompNode):
        self.A = A

    def forward(self):
        self.A.forward()
        self.tensor = np.sqrt(self.A.tensor)

    def backward(self, seed: np.ndarray | float):
        # f = sqrt(A)
        # df/dA = 1/(2sqrt(A)) * dA/dA
        self.A.backward(seed / (2.0 * np.sqrt(self.A.tensor)))