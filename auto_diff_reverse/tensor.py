import numpy as np
from .comp_node import CompNode

class Tensor(CompNode):
    """
    Tensor (Reverse Accumulation)
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

    requires_grad : bool
        Whether this tensor requires gradient computation.\\
        If `True`, gradients will be tracked during the backward pass.\\
        If `False`, this tensor will not accumulate gradients.
    """
    def __init__(self, value: np.ndarray | float, requires_grad: bool = True):
        # only allow numpy arrays or float
        if not isinstance(value, (np.ndarray, float)):
            raise TypeError("Value must be a NumPy array or a float.")
        super().__init__()
        self.tensor = value
        self.partial = None
        self.requires_grad = requires_grad
        if isinstance(self.tensor, np.ndarray):
            self.partial = np.zeros_like(self.tensor, dtype=np.float64)
        else:
            self.partial = 0.0

    @property
    def _signature(self):
        return id(self)

    def assign(self, value: np.ndarray | float):
        if not isinstance(value, (np.ndarray, float)):
            raise TypeError("Assigned value must be a NumPy array or a float.")
        self.tensor = value
        if isinstance(self.tensor, np.ndarray):
            self.partial = np.zeros_like(self.tensor, dtype=np.float64)
        else:
            self.partial = 0.0
        for parent in self.parent_op:
            parent.mark_dirty()

    def forward(self, cc = True):
        if cc: # clear cache flag
            self.clear_graph_cache()
    
    def backward(self, seed: np.ndarray | float):
        if self.requires_grad:
            self.partial = np.add(self.partial, seed)

    @property
    def grad(self):
        """
        Gradient of the tensor with respect to the final output (seed)
    
        This is an alias for the accumulated partial derivatives
        stored during backpropagation.
        """
        return self.partial

    def zero_grad(self):
        """
        Resets the accumulated gradient (partial) to zero.

        This is necessary before performing a new backward pass,
        as gradients are accumulated in reverse-mode auto-diff.
        If not cleared, calling backward() multiple times will 
        result in accumulated gradients from previous passes.
        """
        if isinstance(self.tensor, np.ndarray):
            self.partial = np.zeros_like(self.tensor, dtype=np.float64)
        else:
            self.partial = 0.0