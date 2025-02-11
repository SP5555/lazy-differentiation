from abc import abstractmethod
import numpy as np
from .comp_node import CompNode, GLOBAL_GRAPH_CACHE

# This is some division by zero prevention trickery
# Disabled (permanently? Yes. probably don't need it.)
EPSILON = 0.0

class Operation(CompNode):

    def __new__(cls, *arg, **kwargs):
        sig = cls.signature(*arg, **kwargs)
        if sig in GLOBAL_GRAPH_CACHE:
            GLOBAL_GRAPH_CACHE[sig]._is_repeated = True
            return GLOBAL_GRAPH_CACHE[sig]
        instance = super().__new__(cls)
        instance._cached_value = None
        GLOBAL_GRAPH_CACHE[sig] = instance
        instance._is_repeated = False
        return instance

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    # the true compute call
    def compute_forward(self) -> np.ndarray | float:
        pass

    # forward returns the cached value
    # calls compute_forward() if cached value is not available
    def forward(self) -> np.ndarray | float:
        if self._cached_value is not None:
            return self._cached_value
        self._cached_value = self.compute_forward()
        return self._cached_value
    
    def backward(self) -> np.ndarray | float:
        raise NotImplementedError

class Negate(Operation):

    def __init__(self, A: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A

    @classmethod
    def signature(self, A: CompNode):
        return ("Neg", id(A))
    
    def compute_forward(self) -> np.ndarray | float:
        self.A_tmp = self.A.forward()

        return -1 * self.A_tmp

    def backward(self, w_r_t: str) -> np.ndarray | float:
        return -1 * self.A.backward(w_r_t)

class Add(Operation):

    def __init__(self, A: CompNode, B: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A
        self.B = B
    
    @classmethod
    def signature(self, A: CompNode, B: CompNode):
        return ("Add", id(A), id(B))
    
    def compute_forward(self) -> np.ndarray | float:
        self.A_tmp = self.A.forward()
        self.B_tmp = self.B.forward()

        return self.A_tmp + self.B_tmp
    
    def backward(self, w_r_t: str) -> np.ndarray | float:
        return self.A.backward(w_r_t) + self.B.backward(w_r_t)

class Subtract(Operation):
    
    def __init__(self, A: CompNode, B: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A
        self.B = B

    @classmethod
    def signature(self, A: CompNode, B: CompNode):
        return ("Sub", id(A), id(B))

    def compute_forward(self) -> np.ndarray | float:
        self.A_tmp = self.A.forward()
        self.B_tmp = self.B.forward()

        return self.A_tmp - self.B_tmp
    
    def backward(self, w_r_t: str) -> np.ndarray | float:
        return self.A.backward(w_r_t) - self.B.backward(w_r_t)

class Multiply(Operation):

    def __init__(self, A: CompNode, B: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A
        self.B = B

    @classmethod
    def signature(self, A: CompNode, B: CompNode):
        return ("Mul", id(A), id(B))
    
    def compute_forward(self) -> np.ndarray | float:
        self.A_tmp = self.A.forward()
        self.B_tmp = self.B.forward()

        return self.A_tmp * self.B_tmp
    
    def backward(self, w_r_t: str) -> np.ndarray | float:
        return self.A_tmp * self.B.backward(w_r_t) + self.A.backward(w_r_t) * self.B_tmp

class Divide(Operation):

    def __init__(self, A: CompNode, B: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A
        self.B = B

    @classmethod
    def signature(self, A: CompNode, B: CompNode):
        return ("TDiv", id(A), id(B))
    
    def compute_forward(self) -> np.ndarray | float:
        self.A_tmp = self.A.forward()
        self.B_tmp = self.B.forward()

        return self.A_tmp / self.B_tmp
    
    def backward(self, w_r_t: str) -> np.ndarray | float:
        B_sq = self.B_tmp * self.B_tmp
        return (self.B_tmp * self.A.backward(w_r_t) - self.A_tmp * self.B.backward(w_r_t)) / B_sq
    
# Exponential
class Exp(Operation):

    def __init__(self, A: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A

    @classmethod
    def signature(self, A: CompNode):
        return ("Exp", id(A))

    def compute_forward(self) -> np.ndarray | float:
        self.A_tmp = self.A.forward()

        return np.exp(self.A_tmp)

    def backward(self, w_r_t: str) -> np.ndarray | float:
        return self.forward() * self.A.backward(w_r_t)

# Natural Log
class Log(Operation):

    def __init__(self, A: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A

    @classmethod
    def signature(self, A: CompNode):
        return ("Add", id(A))

    def compute_forward(self) -> np.ndarray | float:
        self.A_tmp = self.A.forward()

        return np.log(self.A_tmp)

    def backward(self, w_r_t: str) -> np.ndarray | float:
        return 1 / self.A_tmp * self.A.backward(w_r_t)

class Power(Operation):

    def __init__(self, B: CompNode, E: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.B = B
        self.E = E

    @classmethod
    def signature(self, A: CompNode, B: CompNode):
        return ("Pow", id(A), id(B))

    def compute_forward(self) -> np.ndarray | float:
        self.B_tmp = self.B.forward()
        self.E_tmp = self.E.forward()

        return np.power(self.B_tmp, self.E_tmp)

    # This is some witchcraft I never learned anywhere.
    # h(x) = f(x)^g(x)
    # h'(x) = f(x)^g(x) (g'(x)ln(f(x)) + g(x)f'(x)/f(x))
    def backward(self, w_r_t: str) -> np.ndarray | float:
        return self.forward() * (self.E.backward(w_r_t) * np.log(self.B_tmp + EPSILON) + self.E_tmp * self.B.backward(w_r_t) / (self.B_tmp + EPSILON))

class Sqrt(Operation):

    def __init__(self, A: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A

    @classmethod
    def signature(self, A: CompNode):
        return ("Add", id(A))
    
    def compute_forward(self) -> np.ndarray | float:
        self.A_tmp = self.A.forward()

        return np.sqrt(self.A_tmp)
    
    def backward(self, w_r_t: str) -> np.ndarray | float:
        return 0.5 / self.forward() * self.A.backward(w_r_t)

class Tanh(Operation):

    def __init__(self, A: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A

    @classmethod
    def signature(self, A: CompNode):
        return ("Tanh", id(A))

    def compute_forward(self) -> np.ndarray | float:
        self.A_tmp = self.A.forward()

        return np.tanh(self.A_tmp)

    def backward(self, w_r_t: str) -> np.ndarray | float:
        return (1 - self.forward() ** 2) * self.A.backward(w_r_t)

class Sigmoid(Operation):

    def __init__(self, A: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A

    @classmethod
    def signature(self, A: CompNode):
        return ("Sigmoid", id(A))
    
    def compute_forward(self) -> np.ndarray | float:
        self.A_tmp = self.A.forward()

        return (np.tanh(self.A_tmp / 2) + 1) / 2
    
    def backward(self, w_r_t: str) -> np.ndarray | float:
        return self.forward() * (1 - self.forward()) * self.A.backward(w_r_t)