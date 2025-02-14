import numpy as np
from .comp_node import CompNode

class Tensor(CompNode):
    """
    Tensor
    =====
    A class representing a tensor in the context of automatic differentiation.

    Parameters
    ----------
    value : np.ndarray, float
        The value assigned to the tensor, used in both
        the forward and backward passes of the computation.
    """
    def __init__(self, value: np.ndarray | float):
        if not isinstance(value, (np.ndarray, float)):
            raise TypeError("Value must be a NumPy array or a float.")
        self.tensor = value
        self.partial = None
        if isinstance(self.tensor, np.ndarray):
            self.partial = np.zeros_like(self.tensor, dtype=np.float64)
        else:
            self.partial = 0.0
    
    def forward(self):
        pass
    
    def backward(self, seed: np.ndarray | float):
        self.partial += seed

    @property
    def grad(self):
        return self.partial