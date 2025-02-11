from .comp_node import CompNode

class Operation(CompNode):

    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError

    def __add__(self, other):
        return Add(self, other)
    
    def __mul__(self, other):
        return Multiply(self, other)
    
    @property
    def value(self):
        return self.forward()

class Add(Operation):
    def __init__(self, A: CompNode, B: CompNode):
        self.A = A
        self.B = B
    
    def forward(self):
        self.data = self.A.forward() + self.B.forward()
        return self.data
    
    def backward(self, w_r_t: str):
        return self.A.backward(w_r_t) + self.B.backward(w_r_t)

class Multiply(Operation):
    def __init__(self, A: CompNode, B: CompNode):
        self.A = A
        self.B = B
    
    def forward(self):
        self.data = self.A.forward() * self.B.forward()
        return self.data
    
    def backward(self, w_r_t: str):
        return self.A.forward() * self.B.backward(w_r_t) + self.A.backward(w_r_t) * self.B.forward()
