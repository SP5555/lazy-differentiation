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
    def backward(self):
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

    def backward(self, seed: np.ndarray | float):
        # f = -A
        # df/dA = -dA/dA
        self.A.backward(-seed)

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

    def backward(self, seed: np.ndarray | float):
        # f = A + B
        # df/dA = dA/dA
        # df/dB = dB/dB
        self.A.backward(seed)
        self.B.backward(seed)

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

    def backward(self, seed: np.ndarray | float):
        # f = A - B
        # df/dA = dA/dA
        # df/dB = -dB/dB
        self.A.backward(seed)
        self.B.backward(-seed)

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

    def backward(self, seed: np.ndarray | float):
        # f = AB
        # df/dA = B * dA/dA
        # df/dB = A * dB/dB
        self.A.backward(self.B.tensor * seed)
        self.B.backward(self.A.tensor * seed)

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

    def backward(self, seed: np.ndarray | float):
        # f = A/B
        # df/dA = 1 / B * dA/dA
        # df/dB = -A / (B^2) * dB/dB
        self.A.backward(seed / self.B.tensor)
        self.B.backward(-self.A.tensor * seed / (self.B.tensor ** 2))

# Exponential
class Exp(Operation):

    def __init__(self, A: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A

    def compute_forward(self):
        self.A.forward(cc=False)
        self.tensor = np.exp(self.A.tensor)

    def backward(self, seed: np.ndarray | float):
        # f = e^A
        # df/dA = e^A * dA/dA
        self.A.backward(self.tensor * seed)

# Natural Log
class Log(Operation):

    def __init__(self, A: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A

    def compute_forward(self):
        self.A.forward(cc=False)
        self.tensor = np.log(self.A.tensor)

    def backward(self, seed: np.ndarray | float):
        # f = ln(A)
        # df/dA = 1 / A * dA/dA
        self.A.backward(seed / self.A.tensor)

class Square(Operation):

    def __init__(self, A: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A

    def compute_forward(self):
        self.A.forward(cc=False)
        self.tensor = self.A.tensor ** 2

    def backward(self, seed: np.ndarray | float):
        # f = A^2
        # df/dA = 2 * A * dA/dA
        self.A.backward(2 * self.tensor * seed)

class Power(Operation):

    def __init__(self, A: CompNode, B: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A
        self.B = B

    def compute_forward(self):
        self.A.forward(cc=False)
        self.B.forward(cc=False)
        self.tensor = self.A.tensor ** self.B.tensor

    def backward(self, seed: np.ndarray | float):
        # f = A^B
        # df/dA = B * A^(B-1) * dA/dA
        # df/dB = ln(A) * A^B * dB/dB
        self.A.backward(self.B.tensor * (self.A.tensor ** (self.B.tensor - 1.0)) * seed)
        self.B.backward(np.log(self.A.tensor) * self.tensor * seed)

class Sqrt(Operation):

    def __init__(self, A: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A

    def compute_forward(self):
        self.A.forward(cc=False)
        self.tensor = np.sqrt(self.A.tensor)

    def backward(self, seed: np.ndarray | float):
        # f = sqrt(A)
        # df/dA = 1 / (2 * sqrt(A)) * dA/dA
        self.A.backward(0.5 * seed / self.tensor)

class Tanh(Operation):

    def __init__(self, A: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A

    def compute_forward(self):
        self.A.forward(cc=False)
        self.tensor = np.tanh(self.A.tensor)

    def backward(self, seed: np.ndarray | float):
        # f = tanh(A)
        # df/dA = (1 - [tanh(A)]^2) * dA/dA
        self.A.backward((1.0 - (self.tensor ** 2)) * seed)

class Sigmoid(Operation):

    def __init__(self, A: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A

    def compute_forward(self):
        self.A.forward(cc=False)
        self.tensor = (np.tanh(self.A.tensor / 2) + 1) / 2

    def backward(self, seed: np.ndarray | float):
        # f = sigmoid(A)
        # df/dA = sigmoid(A) * (1 - sigmoid(A)) * dA/dA
        self.A.backward(self.tensor * (1 - self.tensor) * seed)

# Dot product
class Matmul(Operation):
    def __init__(self, A: CompNode, B: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A
        self.B = B

    def compute_forward(self):
        self.A.forward(cc=False)
        self.B.forward(cc=False)
        self.tensor = np.matmul(self.A.tensor, self.B.tensor)

    def backward(self, seed: np.ndarray | float):
        # Z = A dot B
        # dZ/dA = dZ/dZ dot B^T
        # dZ/dB = A^T dot dZ/dZ
        self.A.backward(np.matmul(seed, self.B.tensor.T))
        self.B.backward(np.matmul(self.A.tensor.T, seed))

class Softmax(Operation):
    # Softmax is one hell of a tricky activation
    # to get the auto-diff system to work with
    # due to its interdependencies between outputs
    def __init__(self, A: CompNode):
        if self._is_repeated: return
        super().__init__()
        self.A = A
    
    def compute_forward(self):
        self.A.forward(cc=False)
        exp = np.exp(self.A.tensor - np.max(self.A.tensor, axis=0, keepdims=True))
        self.tensor = exp / np.sum(exp, axis=0, keepdims=True)
    
    def backward(self, seed: np.ndarray | float):
        # math
        # S_i is softmax(z_i)
        # Jacobian = diag(S) - S dot S.T
        # where each entry is
        # dS_i/dz_j = S_i * (delta_ij - S_j)
        # delta_ij is Kronecker delta term
        # (Simply put, it is an entry in Identity matrix, either 0 or 1)

        # dL/dS = seed
        # dL/dz =  dS/dz dot dL/dS = Jacobian dot seed

        # But, we can avoid constructing Jacobian (which would be a sweet 3D tensor nightmare)
        # each input z's gradient dL/dz_i of dL/dz is given as follows:
        # dL/dz_i = Sum[ S_i * ( delta_ij - S_j ) * dL/dS_j ]  // j goes through all output neurons
        # dL/dz_i = S_i * Sum[ ( delta_ij - S_j ) * dL/dS_j ]  // factor out S_i
        # dL/dz_i = S_i * ( dL/dS_i - Sum[ S_j * dL/dS_j ] )   // break down delta_ij term
        # dL/dz_i = S_i * ( seed_i - Sum[ S_j * seed_j ] )

        # this is softmax
        S = self.tensor

        # this line took years off my lifespan
        dL_dz = S * (seed - np.sum(S * seed, axis = 0, keepdims=True))

        # this backward pass call just accumulates into partials
        # because all calculations are already done inside dL/dz term 
        self.A.backward(dL_dz)