from abc import ABC, abstractmethod

class CompNode(ABC):
    """
    Computational Node
    =====
    """
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self, w_r_t: str):
        pass

    @property
    def value(self):
        return self.forward()

    def __add__(self, other):
        from .operations import Add
        return Add(self, other)

    def __mul__(self, other):
        from .operations import Multiply
        return Multiply(self, other)