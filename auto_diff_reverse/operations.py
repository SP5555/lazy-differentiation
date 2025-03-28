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

    def __init__(self, *args: CompNode, requires_grad: bool = None):
        super().__init__()
        if requires_grad is None:
            self.requires_grad = any(tensor.requires_grad for tensor in args)
        else:
            self.requires_grad = requires_grad
        self._dirty = True

    @classmethod
    def _signature(cls, *args, **kwargs):
        return (id(cls), *(id(arg) for arg in args))

    def mark_dirty(self):
        if not self._dirty:
            self._dirty = True
            for parent in self.parent_op:
                parent.mark_dirty()

    # perform forward pass computation
    # calls compute_forward() if cached tensor is not available
    # forward call auto-clears the global cache
    def forward(self, cc = True):
        if cc: # clear cache flag
            self.clear_graph_cache()
        if self._dirty:
            self._forward_impl()
            self._dirty = False

    def backward(self, seed: np.ndarray | float):
        if self.requires_grad:
            self._backward_impl(seed)

    @abstractmethod
    # computes the forward pass and caches the resulting tensor
    def _forward_impl(self):
        pass

    @abstractmethod
    def _backward_impl(self):
        pass

class Negate(Operation):

    def __init__(self, A: CompNode, requires_grad: bool = None):
        if self._is_repeated: return
        super().__init__(A, requires_grad=requires_grad)
        self.A = A
        self.A.add_parent_op(self)

    def _forward_impl(self):
        self.A.forward(cc=False)
        self.tensor = -self.A.tensor

    def _backward_impl(self, seed: np.ndarray | float):
        # f = -A
        # df/dA = -dA/dA
        self.A.backward(-seed)

class Add(Operation):

    def __init__(self, A: CompNode, B: CompNode, requires_grad: bool = None):
        if self._is_repeated: return
        super().__init__(A, B, requires_grad=requires_grad)
        self.A = A
        self.B = B
        self.A.add_parent_op(self)
        self.B.add_parent_op(self)

    def _forward_impl(self):
        self.A.forward(cc=False)
        self.B.forward(cc=False)
        self.tensor = self.A.tensor + self.B.tensor

    def _backward_impl(self, seed: np.ndarray | float):
        # f = A + B
        # df/dA = dA/dA
        # df/dB = dB/dB
        self.A.backward(seed)
        self.B.backward(seed)

class Subtract(Operation):

    def __init__(self, A: CompNode, B: CompNode, requires_grad: bool = None):
        if self._is_repeated: return
        super().__init__(A, B, requires_grad=requires_grad)
        self.A = A
        self.B = B
        self.A.add_parent_op(self)
        self.B.add_parent_op(self)

    def _forward_impl(self):
        self.A.forward(cc=False)
        self.B.forward(cc=False)
        self.tensor = self.A.tensor - self.B.tensor

    def _backward_impl(self, seed: np.ndarray | float):
        # f = A - B
        # df/dA = dA/dA
        # df/dB = -dB/dB
        self.A.backward(seed)
        self.B.backward(-seed)

class Multiply(Operation):

    def __init__(self, A: CompNode, B: CompNode, requires_grad: bool = None):
        if self._is_repeated: return
        super().__init__(A, B, requires_grad=requires_grad)
        self.A = A
        self.B = B
        self.A.add_parent_op(self)
        self.B.add_parent_op(self)

    def _forward_impl(self):
        self.A.forward(cc=False)
        self.B.forward(cc=False)
        self.tensor = self.A.tensor * self.B.tensor

    def _backward_impl(self, seed: np.ndarray | float):
        # f = AB
        # df/dA = B * dA/dA
        # df/dB = A * dB/dB
        self.A.backward(self.B.tensor * seed)
        self.B.backward(self.A.tensor * seed)

class Divide(Operation):

    def __init__(self, A: CompNode, B: CompNode, requires_grad: bool = None):
        if self._is_repeated: return
        super().__init__(A, B, requires_grad=requires_grad)
        self.A = A
        self.B = B
        self.A.add_parent_op(self)
        self.B.add_parent_op(self)

    def _forward_impl(self):
        self.A.forward(cc=False)
        self.B.forward(cc=False)
        self.tensor = self.A.tensor / self.B.tensor

    def _backward_impl(self, seed: np.ndarray | float):
        # f = A/B
        # df/dA = 1 / B * dA/dA
        # df/dB = -A / (B^2) * dB/dB
        self.A.backward(seed / self.B.tensor)
        self.B.backward(-self.A.tensor * seed / (self.B.tensor ** 2))

class Maximum(Operation):

    def __init__(self, A: CompNode, B: CompNode, requires_grad: bool = None):
        if self._is_repeated: return
        super().__init__(A, B, requires_grad=requires_grad)
        self.A = A
        self.B = B
        self.A.add_parent_op(self)
        self.B.add_parent_op(self)

    def _forward_impl(self):
        self.A.forward(cc=False)
        self.B.forward(cc=False)
        self.tensor = np.maximum(self.A.tensor, self.B.tensor)

    def _backward_impl(self, seed: np.ndarray | float):
        # f = Max(A, B)
        # df/dA = 1 * dA/dA     if A > B
        # df/dA = 0 * dA/dA     if A < B
        # df/dB = 0 * dB/dB     if A > B
        # df/dB = 1 * dB/dB     if A < B
        # if A == B? ehh, impossible but just do
        # df/dA = 0.5 * dA/dA
        # df/dB = 0.5 * dB/dB
        A_g = (self.A.tensor > self.B.tensor) + 0.5 * (self.A.tensor == self.B.tensor)
        B_g = (self.A.tensor < self.B.tensor) + 0.5 * (self.A.tensor == self.B.tensor)
        self.A.backward(A_g * seed)
        self.B.backward(B_g * seed)

class Minimum(Operation):

    def __init__(self, A: CompNode, B: CompNode, requires_grad: bool = None):
        if self._is_repeated: return
        super().__init__(A, B, requires_grad=requires_grad)
        self.A = A
        self.B = B
        self.A.add_parent_op(self)
        self.B.add_parent_op(self)

    def _forward_impl(self):
        self.A.forward(cc=False)
        self.B.forward(cc=False)
        self.tensor = np.minimum(self.A.tensor, self.B.tensor)

    def _backward_impl(self, seed: np.ndarray | float):
        # f = Min(A, B)
        # df/dA = 0 * dA/dA     if A > B
        # df/dA = 1 * dA/dA     if A < B
        # df/dB = 1 * dB/dB     if A > B
        # df/dB = 0 * dB/dB     if A < B
        # if A == B? again, just do
        # df/dA = 0.5 * dA/dA
        # df/dB = 0.5 * dB/dB
        A_g = (self.A.tensor < self.B.tensor) + 0.5 * (self.A.tensor == self.B.tensor)
        B_g = (self.A.tensor > self.B.tensor) + 0.5 * (self.A.tensor == self.B.tensor)
        self.A.backward(A_g * seed)
        self.B.backward(B_g * seed)

class Abs(Operation):

    def __init__(self, A: CompNode, requires_grad: bool = None):
        if self._is_repeated: return
        super().__init__(A, requires_grad=requires_grad)
        self.A = A
        self.A.add_parent_op(self)
    
    def _forward_impl(self):
        self.A.forward(cc=False)
        self.tensor = np.abs(self.A.tensor)

    def _backward_impl(self, seed: np.ndarray | float):
        # f = Abs(A)
        # df/dA =  1 * dA/dA    if A > 0
        # df/dA = -1 * dA/dA    if A < 0
        # if A == 0? I don't know man, do zero
        # df/dA = 0 * dA/dA
        grad_A = np.sign(self.A.tensor)
        grad_A[self.A.tensor == 0] = 0
        self.A.backward(grad_A * seed)

class Clip(Operation):

    def __init__(self, A: CompNode, min: float, max: float, requires_grad: bool = None):
        if self._is_repeated: return
        super().__init__(A, requires_grad=requires_grad)
        self.A = A
        self.min = min
        self.max = max
        self.A.add_parent_op(self)

    def _forward_impl(self):
        self.A.forward(cc=False)
        self.tensor = np.clip(self.A.tensor, self.min, self.max)

    def _backward_impl(self, seed: np.ndarray | float):
        # Gradient is zero where the tensor is clipped (outside the range)
        grad = np.ones_like(self.A.tensor)
        grad[self.A.tensor < self.min] = 0
        grad[self.A.tensor > self.max] = 0
        self.A.backward(grad * seed)

class Mean(Operation):

    def __init__(self,
                 A: CompNode,
                 axis=None,
                 keepdims=False,
                 requires_grad=None):
        if self._is_repeated: return
        super().__init__(A, requires_grad=requires_grad)
        self.A = A
        self.axis = axis
        self.keepdims = keepdims
        self.A.add_parent_op(self)

    def _forward_impl(self):
        self.A.forward(cc=False)
        self.tensor = np.mean(self.A.tensor, axis=self.axis, keepdims=self.keepdims)

    def _backward_impl(self, seed: np.ndarray | float):
        # f = Mean(A) = 1/n * sum [ A_i ]
        # df/dA_j = 1/n * dA_j/dA_j
        n = np.prod(self.A.tensor.shape) if self.axis is None else self.A.tensor.shape[self.axis]
        self.A.backward(np.ones_like(self.A.tensor) / n * seed)

class Variance(Operation):

    def __init__(self,
                 A: CompNode,
                 axis=None,
                 keepdims=False,
                 requires_grad=None):
        if self._is_repeated: return
        super().__init__(A, requires_grad=requires_grad)
        self.A = A
        self.axis = axis
        self.keepdims = keepdims
        self.A.add_parent_op(self)

    def _forward_impl(self):
        self.A.forward(cc=False)
        self.mean = np.mean(self.A.tensor, axis=self.axis, keepdims=self.keepdims)
        self.tensor = np.mean((self.A.tensor - self.mean) ** 2, axis=self.axis, keepdims=self.keepdims)

    def _backward_impl(self, seed: np.ndarray | float):
        # A little brain damage
        # f = Var(A) = 1/n * sum [ (A_i - Mean(A))^2 ]
        #            = 1/n * sum [ A_i^2 - 2 * A_i * Mean(A) + Mean(A)^2 ]
        #            = 1/n * sum [ A_i^2 ] - 2/n * Mean(A) * sum [ A_i ] + Mean(A)^2
        #            = 1/n * sum [ A_i^2 ] - Mean(A)^2
        # df/dA_j = 1/n * 2 * A_j * dA_j/dA_j - 2   * Mean(A) * dMean(A)/dA_j * dA_j/dA_j
        #         = 2/n *     A_j * dA_j/dA_j - 2/n * Mean(A)                 * dA_j/dA_j
        #         = 2/n * ( A_j - Mean(A) ) * dA_j/dA_j
        n = np.prod(self.A.tensor.shape) if self.axis is None else self.A.tensor.shape[self.axis]
        self.A.backward(2 / n * (self.A.tensor - self.mean) * seed)

# Exponential
class Exp(Operation):

    def __init__(self, A: CompNode, requires_grad: bool = None):
        if self._is_repeated: return
        super().__init__(A, requires_grad=requires_grad)
        self.A = A
        self.A.add_parent_op(self)

    def _forward_impl(self):
        self.A.forward(cc=False)
        self.tensor = np.exp(self.A.tensor)

    def _backward_impl(self, seed: np.ndarray | float):
        # f = e^A
        # df/dA = e^A * dA/dA
        self.A.backward(self.tensor * seed)

# Natural Log
class Log(Operation):

    def __init__(self, A: CompNode, requires_grad: bool = None):
        if self._is_repeated: return
        super().__init__(A, requires_grad=requires_grad)
        self.A = A
        self.A.add_parent_op(self)

    def _forward_impl(self):
        self.A.forward(cc=False)
        self.tensor = np.log(self.A.tensor)

    def _backward_impl(self, seed: np.ndarray | float):
        # f = ln(A)
        # df/dA = 1 / A * dA/dA
        self.A.backward(seed / self.A.tensor)

class Square(Operation):

    def __init__(self, A: CompNode, requires_grad: bool = None):
        if self._is_repeated: return
        super().__init__(A, requires_grad=requires_grad)
        self.A = A
        self.A.add_parent_op(self)

    def _forward_impl(self):
        self.A.forward(cc=False)
        self.tensor = self.A.tensor ** 2

    def _backward_impl(self, seed: np.ndarray | float):
        # f = A^2
        # df/dA = 2 * A * dA/dA
        self.A.backward(2 * self.A.tensor * seed)

class Power(Operation):

    def __init__(self, A: CompNode, B: CompNode, requires_grad: bool = None):
        if self._is_repeated: return
        super().__init__(A, B, requires_grad=requires_grad)
        self.A = A
        self.B = B
        self.A.add_parent_op(self)
        self.B.add_parent_op(self)

    def _forward_impl(self):
        self.A.forward(cc=False)
        self.B.forward(cc=False)
        self.tensor = self.A.tensor ** self.B.tensor

    def _backward_impl(self, seed: np.ndarray | float):
        # f = A^B
        # df/dA = B * A^(B-1) * dA/dA
        # df/dB = ln(A) * A^B * dB/dB
        self.A.backward(self.B.tensor * (self.A.tensor ** (self.B.tensor - 1.0)) * seed)
        self.B.backward(np.log(self.A.tensor) * self.tensor * seed)

class Sqrt(Operation):

    def __init__(self, A: CompNode, requires_grad: bool = None):
        if self._is_repeated: return
        super().__init__(A, requires_grad=requires_grad)
        self.A = A
        self.A.add_parent_op(self)

    def _forward_impl(self):
        self.A.forward(cc=False)
        self.tensor = np.sqrt(self.A.tensor)

    def _backward_impl(self, seed: np.ndarray | float):
        # f = sqrt(A)
        # df/dA = 1 / (2 * sqrt(A)) * dA/dA
        self.A.backward(0.5 * seed / self.tensor)

class Tanh(Operation):

    def __init__(self, A: CompNode, requires_grad: bool = None):
        if self._is_repeated: return
        super().__init__(A, requires_grad=requires_grad)
        self.A = A
        self.A.add_parent_op(self)

    def _forward_impl(self):
        self.A.forward(cc=False)
        self.tensor = np.tanh(self.A.tensor)

    def _backward_impl(self, seed: np.ndarray | float):
        # f = tanh(A)
        # df/dA = (1 - [tanh(A)]^2) * dA/dA
        self.A.backward((1.0 - (self.tensor ** 2)) * seed)

class Sigmoid(Operation):

    def __init__(self, A: CompNode, requires_grad: bool = None):
        if self._is_repeated: return
        super().__init__(A, requires_grad=requires_grad)
        self.A = A
        self.A.add_parent_op(self)

    def _forward_impl(self):
        self.A.forward(cc=False)
        self.tensor = (np.tanh(self.A.tensor / 2) + 1) / 2

    def _backward_impl(self, seed: np.ndarray | float):
        # f = sigmoid(A)
        # df/dA = sigmoid(A) * (1 - sigmoid(A)) * dA/dA
        self.A.backward(self.tensor * (1 - self.tensor) * seed)

# Dot product
class Matmul(Operation):
    def __init__(self, A: CompNode, B: CompNode, requires_grad: bool = None):
        if self._is_repeated: return
        super().__init__(A, B, requires_grad=requires_grad)
        self.A = A
        self.B = B
        self.A.add_parent_op(self)
        self.B.add_parent_op(self)

    def _forward_impl(self):
        self.A.forward(cc=False)
        self.B.forward(cc=False)
        self.tensor = np.matmul(self.A.tensor, self.B.tensor)

    def _backward_impl(self, seed: np.ndarray | float):
        # Z = A dot B
        # dZ/dA = dZ/dZ dot B^T
        # dZ/dB = A^T dot dZ/dZ
        self.A.backward(np.matmul(seed, self.B.tensor.T))
        self.B.backward(np.matmul(self.A.tensor.T, seed))

class Softmax(Operation):
    # Softmax is one hell of a tricky activation
    # to get the auto-diff system to work with
    # due to its interdependencies between outputs
    def __init__(self, A: CompNode, requires_grad: bool = None):
        if self._is_repeated: return
        super().__init__(A, requires_grad=requires_grad)
        self.A = A
        self.A.add_parent_op(self)
    
    def _forward_impl(self):
        self.A.forward(cc=False)
        exp = np.exp(self.A.tensor - np.max(self.A.tensor, axis=0, keepdims=True))
        self.tensor = exp / np.sum(exp, axis=0, keepdims=True)
    
    def _backward_impl(self, seed: np.ndarray | float):
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

        # this line took years off my lifespan 
        self.A.backward(self.tensor * (seed - np.sum(self.tensor * seed, axis = 0, keepdims=True)))

class Huber(Operation):

    def __init__(self, A: CompNode, d: float, requires_grad: bool = None):
        if self._is_repeated: return
        super().__init__(A, requires_grad=requires_grad)
        self.A = A
        self.d = d
        self.A.add_parent_op(self)
    
    def _forward_impl(self):
        self.A.forward(cc=False)

        # masks
        mid = np.abs(self.A.tensor) <= self.d
        pos = self.A.tensor > self.d
        neg = self.A.tensor < -self.d

        mid_expression = (0.5 * (self.A.tensor ** 2) * mid)
        p_expression = self.d * (self.A.tensor - self.d/2) * pos
        n_expression = self.d * (-self.A.tensor - self.d/2) * neg

        self.tensor = mid_expression + p_expression + n_expression
    
    def _backward_impl(self, seed: np.ndarray | float):
        # f = 1/2 * a^2         if |a| <= d
        # f = d * (|a| - d/2)   if |a| > d
        # df/dA = a * dA/dA     if |a| <= d
        # df/dA =  d * dA/dA    if a > d
        # df/dA = -d * dA/dA    if a < -d
        mid_grad = self.A.tensor * (np.abs(self.A.tensor) <= self.d)
        p_grad = self.d * (self.A.tensor > self.d)
        n_grad = -self.d * (self.A.tensor < -self.d)

        grad_A = mid_grad + p_grad + n_grad
        self.A.backward(grad_A * seed)