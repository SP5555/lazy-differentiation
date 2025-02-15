import sys
import os

# some issues
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from auto_diff_forward import *
from auto_diff_forward.comp_node import GLOBAL_GRAPH_CACHE

def main():

    a = Tensor(np.array([1, 2, 3]), "a")
    b = Tensor(np.array([4, 5, 6]), "b")
    c = Tensor(np.array([7, 8, 9]), "c")
    two = Tensor(2.0) # wrap inside Tensor if you wanna use a constant

    # expression = (a + b) ** (a / b)
    # expression = Sigmoid(a/b) - Exp(b) + Square(a)
    # expression = Log(b) * Tanh(Sqrt(a/b)) + a ** Sqrt(two * b)
    expression = (a*b) + (a*b) * (a*b)
    # expression = a

    expression.forward()

    np.set_printoptions(precision=8)

    print(f"Forward: {expression.evaluate()}")
    print(f"df/da  : {expression.backward('a')}")
    print(f"df/db  : {expression.backward('b')}")
    print(f"df/dc  : {expression.backward('c')}")

    # cached values should be empty as forward() clears cache
    print("global cache")
    for k in GLOBAL_GRAPH_CACHE:
        print(k)

if __name__ == "__main__":
    main()