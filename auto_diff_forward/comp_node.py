from abc import ABC, abstractmethod
from numpy import ndarray

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .operations import Operation

GLOBAL_GRAPH_CACHE = {}

class CompNode(ABC):
    """
    Computational Node
    =====
    """
    @abstractmethod
    # returns a Unique signature to be stored in graph cache
    def _signature(self):
        pass

    @staticmethod
    def clear_graph_cache():
        GLOBAL_GRAPH_CACHE.clear()

    def __init__(self):
        self.tensor: ndarray | float = None
        self.parent_op: set["Operation"] = set()

    def add_parent_op(self, node: "Operation"):
        self.parent_op.add(node)

    @abstractmethod
    def forward(self):
        """
        Forward Pass
        =====

        This method is responsible for performing the forward pass computation. \\
        Call `evaluate()` to retrive the calculated value.
        """
        pass

    @abstractmethod
    def backward(self, w_r_t: str) -> ndarray | float:
        """
        Backward Pass
        =====

        This method calculates the gradients during the backward pass.
        It takes the name of the variable with respect to which the
        derivative is being computed (w.r.t) and returns the computed gradient.

        Parameters
        ----------
        w_r_t : str
            The name of the variable with respect to which the derivative is computed.

        Returns
        -------
        ndarray or float
            The gradient of the operation with respect to the input(s) in the operation,
            typically a scalar or an array representing the computed gradient.
        """
        pass

    def evaluate(self) -> ndarray | float:
        """
        Evaluate
        =====

        This method can be called only after forward pass.

        Returns
        -------
        np.ndarray | float
            The value of the expression computed by forward pass.
        """
        return self.tensor

    def __neg__(self):
        from .operations import Negate
        return Negate(self)

    def __add__(self, other):
        from .operations import Add
        return Add(self, other)

    def __sub__(self, other):
        from .operations import Subtract
        return Subtract(self, other)

    def __mul__(self, other):
        from .operations import Multiply
        return Multiply(self, other)

    def __truediv__(self, other):
        from .operations import Divide
        return Divide(self, other)
    
    def __pow__(self, other):
        from .operations import Power
        return Power(self, other)