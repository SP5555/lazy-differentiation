from abc import abstractmethod
import numpy as np
from .comp_node import CompNode, GLOBAL_GRAPH_CACHE

class Operation(CompNode):

    def __new__(cls, *arg, **kwargs):
        sig = cls._signature(*arg, **kwargs)
        if sig in GLOBAL_GRAPH_CACHE:
            GLOBAL_GRAPH_CACHE[sig]._is_repeated = True
            return GLOBAL_GRAPH_CACHE[sig]
        instance = super().__new__(cls)
        GLOBAL_GRAPH_CACHE[sig] = instance
        instance._is_repeated = False
        return instance

    def __init__(self):
        super().__init__()

    @classmethod
    def _signature(cls, *args):
        return (id(cls), *(id(arg) for arg in args))

    # perform forward pass computation
    # calls compute_forward() if cached tensor is not available
    # forward call auto-clears the global cache
    def forward(self, cc = True):
        if cc: # clear cache flag
            self.clear_graph_cache()
        if self.tensor is None:
            self.compute_forward()

    @abstractmethod
    def backward(self, w_r_t: str) -> np.ndarray | float:
        pass

    @abstractmethod
    # computes the forward pass and caches the resulting tensor
    def compute_forward(self):
        pass

class Negate(Operation):

    def __init__(self, A: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A
    
    def compute_forward(self):
        self.A.forward(cc=False)
        self.tensor = -self.A.tensor

    def backward(self, w_r_t: str) -> np.ndarray | float:
        # h(x) = -f(x)
        # h'(x) = -f'(x)
        return -self.A.backward(w_r_t)

class Add(Operation):

    def __init__(self, A: CompNode, B: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A
        self.B = B
    
    def compute_forward(self):
        self.A.forward(cc=False)
        self.B.forward(cc=False)
        self.tensor = self.A.tensor + self.B.tensor

    def backward(self, w_r_t: str) -> np.ndarray | float:
        # h(x) = f(x) + g(x)
        # h'(x) = f'(x) + g'(x)
        return self.A.backward(w_r_t) + self.B.backward(w_r_t)

class Subtract(Operation):
    
    def __init__(self, A: CompNode, B: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A
        self.B = B

    def compute_forward(self):
        self.A.forward(cc=False)
        self.B.forward(cc=False)
        self.tensor = self.A.tensor - self.B.tensor

    def backward(self, w_r_t: str) -> np.ndarray | float:
        # h(x) = f(x) - g(x)
        # h'(x) = f'(x) - g'(x)
        return self.A.backward(w_r_t) - self.B.backward(w_r_t)

class Multiply(Operation):

    def __init__(self, A: CompNode, B: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A
        self.B = B

    def compute_forward(self):
        self.A.forward(cc=False)
        self.B.forward(cc=False)
        self.tensor = self.A.tensor * self.B.tensor

    def backward(self, w_r_t: str) -> np.ndarray | float:
        # h(x) = f(x)g(x)
        # h'(x) = f(x)g'(x) + f'(x)g(x)
        return self.A.tensor * self.B.backward(w_r_t) + self.A.backward(w_r_t) * self.B.tensor

class Divide(Operation):

    def __init__(self, A: CompNode, B: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A
        self.B = B

    def compute_forward(self):
        self.A.forward(cc=False)
        self.B.forward(cc=False)
        self.tensor = self.A.tensor / self.B.tensor

    def backward(self, w_r_t: str) -> np.ndarray | float:
        # h(x) = f(x)/g(x)
        # h'(x) = [g(x)f'(x) - f(x)g'(x)] / (g(x)^2)
        B_sq = self.B.tensor ** 2
        return (self.B.tensor * self.A.backward(w_r_t) - self.A.tensor * self.B.backward(w_r_t)) / B_sq
    
# Exponential
class Exp(Operation):

    def __init__(self, A: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A

    def compute_forward(self):
        self.A.forward(cc=False)
        self.tensor = np.exp(self.A.tensor)

    def backward(self, w_r_t: str) -> np.ndarray | float:
        # h(x) = e^f(x)
        # h'(x) = e^f(x) * f'(x)
        return self.tensor * self.A.backward(w_r_t)

# Natural Log
class Log(Operation):

    def __init__(self, A: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A

    def compute_forward(self):
        self.A.forward(cc=False)
        self.tensor = np.log(self.A.tensor)

    def backward(self, w_r_t: str) -> np.ndarray | float:
        # h(x) = ln(f(x))
        # h'(x) = 1 / f(x) * f'(x)
        return self.A.backward(w_r_t) / self.A.tensor

class Square(Operation):

    def __init__(self, A: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A

    def compute_forward(self):
        self.A.forward(cc=False)
        self.tensor = self.A.tensor ** 2

    def backward(self, w_r_t: str) -> np.ndarray | float:
        # h(x) = f(x)^2
        # h'(x) = 2 * f(x) * f'(x)
        return 2 * self.A.tensor * self.A.backward(w_r_t)

class Power(Operation):

    def __init__(self, B: CompNode, E: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.B = B
        self.E = E

    def compute_forward(self):
        self.B.forward(cc=False)
        self.E.forward(cc=False)
        self.tensor = self.B.tensor ** self.E.tensor

    def backward(self, w_r_t: str) -> np.ndarray | float:
        # This is some witchcraft I never learned anywhere.
        # h(x) = f(x)^g(x)
        # h'(x) = f(x)^g(x) (g'(x)ln(f(x)) + g(x)f'(x)/f(x))
        return self.tensor * (self.E.backward(w_r_t) * np.log(self.B.tensor) + self.E.tensor * self.B.backward(w_r_t) / (self.B.tensor))

class Sqrt(Operation):

    def __init__(self, A: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A

    def compute_forward(self):
        self.A.forward(cc=False)
        self.tensor = np.sqrt(self.A.tensor)
    
    def backward(self, w_r_t: str) -> np.ndarray | float:
        # h(x) = sqrt(f(x))
        # h'(x) = 1 / (2 * sqrt(f(x))) * f'(x)
        return 0.5 * self.A.backward(w_r_t) / self.tensor

class Tanh(Operation):

    def __init__(self, A: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A

    def compute_forward(self):
        self.A.forward(cc=False)
        self.tensor = np.tanh(self.A.tensor)

    def backward(self, w_r_t: str) -> np.ndarray | float:
        # h(x) = tanh(f(x))
        # h'(x) = (1 - [tanh(f(x))]^2) * f'(x)
        return (1 - self.tensor ** 2) * self.A.backward(w_r_t)

class Sigmoid(Operation):

    def __init__(self, A: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A
    
    def compute_forward(self):
        self.A.forward(cc=False)
        self.tensor = (np.tanh(self.A.tensor / 2) + 1) / 2
    
    def backward(self, w_r_t: str) -> np.ndarray | float:
        # h(x) = sigmoid(f(x))
        # h'(x) = sigmoid(f(x)) * (1 - sigmoid(f(x))) * f'(x)
        return self.tensor * (1 - self.tensor) * self.A.backward(w_r_t)