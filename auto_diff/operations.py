import numpy as np
from .comp_node import CompNode

# This is some division by zero prevention trickery
# Disabled (permanently? Yes. probably don't need it.)
EPSILON = 0.0

class Operation(CompNode):

    def forward(self) -> np.ndarray | float:
        raise NotImplementedError
    
    def backward(self) -> np.ndarray | float:
        raise NotImplementedError
    
    @property
    def value(self):
        return self.forward()

class Negate(Operation):

    def __init__(self, A: CompNode):
        self.A = A
    
    def forward(self) -> np.ndarray | float:
        self.A_tmp = self.A.forward()

        self.val = -1 * self.A_tmp
        return self.val

    def backward(self, w_r_t: str) -> np.ndarray | float:
        return -1 * self.A.backward(w_r_t)

class Add(Operation):

    def __init__(self, A: CompNode, B: CompNode):
        self.A = A
        self.B = B
    
    def forward(self) -> np.ndarray | float:
        self.A_tmp = self.A.forward()
        self.B_tmp = self.B.forward()

        self.val = self.A_tmp + self.B_tmp
        return self.val
    
    def backward(self, w_r_t: str) -> np.ndarray | float:
        return self.A.backward(w_r_t) + self.B.backward(w_r_t)

class Subtract(Operation):
    
    def __init__(self, A: CompNode, B: CompNode):
        self.A = A
        self.B = B

    def forward(self) -> np.ndarray | float:
        self.A_tmp = self.A.forward()
        self.B_tmp = self.B.forward()

        self.val = self.A_tmp - self.B_tmp
        return self.val
    
    def backward(self, w_r_t: str) -> np.ndarray | float:
        return self.A.backward(w_r_t) - self.B.backward(w_r_t)

class Multiply(Operation):

    def __init__(self, A: CompNode, B: CompNode):
        self.A = A
        self.B = B
    
    def forward(self) -> np.ndarray | float:
        self.A_tmp = self.A.forward()
        self.B_tmp = self.B.forward()

        self.val = self.A_tmp * self.B_tmp
        return self.val
    
    def backward(self, w_r_t: str) -> np.ndarray | float:
        return self.A_tmp * self.B.backward(w_r_t) + self.A.backward(w_r_t) * self.B_tmp

class Divide(Operation):

    def __init__(self, A: CompNode, B: CompNode):
        self.A = A
        self.B = B
    
    def forward(self) -> np.ndarray | float:
        self.A_tmp = self.A.forward()
        self.B_tmp = self.B.forward()

        self.val = self.A_tmp / self.B_tmp
        return self.val
    
    def backward(self, w_r_t: str) -> np.ndarray | float:
        B_sq = self.B_tmp * self.B_tmp
        return (self.B_tmp * self.A.backward(w_r_t) - self.A_tmp * self.B.backward(w_r_t)) / B_sq
    
# Exponential
class Exp(Operation):

    def __init__(self, A: CompNode):
        self.A = A

    def forward(self) -> np.ndarray | float:
        self.A_tmp = self.A.forward()

        self.val = np.exp(self.A_tmp)
        return self.val

    def backward(self, w_r_t: str) -> np.ndarray | float:
        return self.val * self.A.backward(w_r_t)

# Natural Log
class Log(Operation):

    def __init__(self, A: CompNode):
        self.A = A

    def forward(self) -> np.ndarray | float:
        self.A_tmp = self.A.forward()

        self.val = np.log(self.A_tmp)
        return self.val

    def backward(self, w_r_t: str) -> np.ndarray | float:
        return 1 / self.A_tmp * self.A.backward(w_r_t)

class Power(Operation):

    def __init__(self, B: CompNode, E: CompNode):
        self.B = B
        self.E = E

    def forward(self) -> np.ndarray | float:
        self.B_tmp = self.B.forward()
        self.E_tmp = self.E.forward()

        self.val = np.power(self.B_tmp, self.E_tmp)
        return self.val

    # This is some witchcraft I never learned anywhere.
    # h(x) = f(x)^g(x)
    # h'(x) = f(x)^g(x) (g'(x)ln(f(x)) + g(x)f'(x)/f(x))
    def backward(self, w_r_t: str) -> np.ndarray | float:
        return self.val * (self.E.backward(w_r_t) * np.log(self.B_tmp + EPSILON) + self.E_tmp * self.B.backward(w_r_t) / (self.B_tmp + EPSILON))

class Sqrt(Operation):

    def __init__(self, A: CompNode):
        self.A = A
    
    def forward(self) -> np.ndarray | float:
        self.A_tmp = self.A.forward()

        self.val = np.sqrt(self.A_tmp)
        return self.val
    
    def backward(self, w_r_t: str) -> np.ndarray | float:
        return 0.5 / self.val * self.A.backward(w_r_t)

class Tanh(Operation):

    def __init__(self, A: CompNode):
        self.A = A

    def forward(self) -> np.ndarray | float:
        self.A_tmp = self.A.forward()

        self.val = np.tanh(self.A_tmp)
        return self.val

    def backward(self, w_r_t: str) -> np.ndarray | float:
        return (1 - self.val ** 2) * self.A.backward(w_r_t)

class Sigmoid(Operation):

    def __init__(self, A: CompNode):
        self.A = A
    
    def forward(self) -> np.ndarray | float:
        self.A_tmp = self.A.forward()

        self.val = (np.tanh(self.A_tmp / 2) + 1) / 2
        return self.val
    
    def backward(self, w_r_t: str) -> np.ndarray | float:
        return self.val * (1 - self.val) * self.A.backward(w_r_t)