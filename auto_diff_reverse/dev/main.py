import sys
import os

# some issues
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from auto_diff_reverse import *
from auto_diff_reverse.comp_node import GLOBAL_GRAPH_CACHE

def main():

    a = Tensor(np.array([1, 2, 3]))
    b = Tensor(np.array([4, 5, 6]))
    c = Tensor(np.array([7, 8, 9]))
    two = Tensor(2.0, requires_grad=False) # wrap inside Tensor if you wanna use a constant
    three = Tensor(3.0, requires_grad=False)

    # neural network layer dummy
    # Ws = Tensor(np.array([[1, -2, 3], [-4, 5, -6], [-1, 2, 3], [4, 5, -6]]))
    # As = Tensor(np.array([[0.2], [0.0], [0.9]]))
    # Bs = Tensor(np.array([[1], [-2], [2], [-1]]))

    # expression = (a + b) ** (a / b)
    # expression = Sigmoid(a/b) - Exp(b) + Square(a)
    expression = Log(b) * Tanh(Sqrt(a/b)) + a ** Sqrt(two * b)
    # expression = (a*b) + (a*b) * (a*b)
    # expression = two * three + a

    # you should broadcast Bs manually
    # don't trust Numpy's auto broadcast feature Lol
    # but yeah, this example doesn't need broadcasting
    # expression = Sigmoid(Matmul(Ws, As) + Bs)

    expression.forward()
    # seed is gradient of loss
    expression.backward(seed=np.array([1, 1, 1]))
    # expression.backward(seed=np.array([[-0.4], [0.0], [-0.6], [0.6]]))

    np.set_printoptions(precision=8)

    print(f"Forward:\n{expression.evaluate()}")
    print(f"df/da  :\n{a.grad}")
    print(f"df/db  :\n{b.grad}")
    print(f"df/dc  :\n{c.grad}")
    # print(f"df/dW:\n{Ws.grad}")
    # print(f"df/dA:\n{As.grad}")
    # print(f"df/dB:\n{Bs.grad}")

    a.zero_grad()
    b.zero_grad()
    c.zero_grad()

    # cached values should be empty as forward() clears cache
    print("global cache")
    for k in GLOBAL_GRAPH_CACHE:
        print(k)

if __name__ == "__main__":
    main()