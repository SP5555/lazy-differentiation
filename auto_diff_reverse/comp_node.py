from abc import ABC, abstractmethod
from numpy import ndarray

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

    def clear_graph_cache(self):
        GLOBAL_GRAPH_CACHE.clear()

    def __init__(self):
        self.tensor = None

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
    def backward(self, seed: ndarray | float):
        """
        Backward Pass
        =====

        This method calculates the gradients during the backward pass
        and updates the partials of base tensors with the computed gradients.

        Parameters
        ----------
        seed : np.ndarray | float
            The gradient of loss used to propagate backwards.
            Set seed to `1` or numpy array of `1` with equivalent
            dimensions for normal gradient calculation.
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