import numpy as np
from .comp_node import CompNode, GLOBAL_GRAPH_CACHE

class Tensor(CompNode):
    """
    Tensor
    =====
    A class representing a tensor in the context of automatic differentiation.

    This class supports the following overloaded operations:
    - Addition (`+`)
    - Negation (`-`)
    - Subtraction (`-`)
    - Multiplication (`*`)
    - Division (`/`)
    - Exponentiation (`**`)

    Parameters
    ----------
    value : np.ndarray, float
        The value assigned to the tensor, used in both
        the forward and backward passes of the computation.

    name : str, optional
        A label for the tensor, used when differentiating with respect to this tensor. \\
        For constants, this parameter is not required.
    """
    def __init__(self, value: np.ndarray | float, name: str = None):
        # Only allow numpy arrays or float
        if not isinstance(value, (np.ndarray, float)):
            raise TypeError("Value must be a NumPy array or a float.")
        self.tensor = value
        self.name = name
    
    @property
    def signature(self):
        return id(self)

    def forward(self, cc: bool) -> np.ndarray | float:
        if cc: # clear cache flag
            self.clear_cache()
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
