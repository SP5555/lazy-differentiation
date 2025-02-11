import numpy as np

class Variable:
    def __init__(self, value: int | np.ndarray, name: str = None):
        self.data = value
        self.name = name

    def forward(self):
        return self.data

    def backward(self, w_r_t: str):
        if w_r_t == self.name:
            return 1
        else:
            return 0

    def __add__(self, other):
        from .operations import Add
        return Add(self, other)
    
    def __mul__(self, other):
        from .operations import Multiply
        return Multiply(self, other)
    
    @property
    def value(self):
        return self.data