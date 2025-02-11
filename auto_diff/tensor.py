import numpy as np
from .comp_node import CompNode

class Tensor(CompNode):

    def __init__(self, value: np.ndarray | float, name: str = None):
        # Only allow numpy arrays or float
        if not isinstance(value, (np.ndarray, float)):
            raise TypeError("Value must be a NumPy array or a float.")
        self.tensor = value
        self.name = name

    def forward(self) -> np.ndarray | float:
        return self.tensor

    def backward(self, w_r_t: str) -> np.ndarray | float:
        if w_r_t == self.name:
            if isinstance(self.tensor, np.ndarray):
                return np.ones_like(self.tensor)
            return 1.0
        else:
            if isinstance(self.tensor, np.ndarray):
                return np.zeros_like(self.tensor)
            return 0.0
