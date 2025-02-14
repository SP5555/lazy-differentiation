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
    def signature(self):
        pass

    def clear_cache(self):
        GLOBAL_GRAPH_CACHE.clear()

    @abstractmethod
    def forward(self) -> ndarray | float:
        """
        Forward Pass
        =====

        This method is responsible for performing the forward pass computation.
        It takes the inputs to the operation (Tensors or Operations) and returns
        the resulting value.

        Returns
        -------
        ndarray or float
            The result of the forward pass computation, typically a scalar
            or an array representing the output of the operation.
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

    @property
    def value(self):
        return self.forward()

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