import numpy as np
from .comp_node import CompNode

class Operation(CompNode):

    def forward(self) -> np.ndarray | float:
        raise NotImplementedError
    
    def backward(self) -> np.ndarray | float:
        raise NotImplementedError

    def __add__(self, other):
        return Add(self, other)
    
    def __mul__(self, other):
        return Multiply(self, other)
    
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
        raise NotImplementedError("Don't use power, do something else.")
    
    def forward(self) -> np.ndarray | float:
        self.B_tmp = self.B.forward()
        self.E_tmp = self.E.forward()

        self.val = np.power(self.B_tmp, self.E_tmp)
        return self.val

    def backward(self, w_r_t: str) -> np.ndarray | float:
        pass # uhhhhh