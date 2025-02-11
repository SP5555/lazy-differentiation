import numpy as np
from .comp_node import CompNode

class Tensor(CompNode):

    def __init__(self, value: np.ndarray | float, name: str = None):
        if not isinstance(value, (np.ndarray, float)):  # Only allow numpy arrays or int
            raise TypeError("Value must be a NumPy array or an integer.")
        self.tensor = value
        self.name = name

    def forward(self):
        return self.tensor

    def backward(self, w_r_t: str):
        if w_r_t == self.name:
            if isinstance(self.tensor, np.ndarray):
                return np.ones_like(self.tensor)
            return 1
        else:
            if isinstance(self.tensor, np.ndarray):
                return np.zeros_like(self.tensor)
            return 0
