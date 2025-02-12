import sys
import os

# some issues
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from auto_diff import *
from auto_diff.comp_node import GLOBAL_GRAPH_CACHE

def main():

    a = Tensor(np.array([1, 2, 3]), "a")
    b = Tensor(np.array([4, 5, 6]), "b")
    c = Tensor(np.array([7, 8, 9]), "c")
    two = Tensor(2.0) # wrap inside Tensor if you wanna use a constant

    # expression = (a + b) ** (a / b)
    # expression = Sigmoid(a/b)
    # expression = Log(b) * Tanh(Sqrt(a/b)) + a ** Sqrt(two * b)
    expression = (a*b) + (a*b) + (a*b)
    # expression = Tanh(a)

    np.set_printoptions(precision=8)

    # forward pass (Grade 1 math)
    print(f"Forward: {expression.forward()}")

    # Easy Lazy Derivatives

    # derivative of expression with respect to "a"
    print(f"df/da  : {expression.backward('a')}")

    # derivative of expression with respect to "b"
    print(f"df/db  : {expression.backward('b')}")
    
    # derivative of expression with respect to "c"
    # (which doesn’t exist, so should be all zeroes)
    print(f"df/dc  : {expression.backward('c')}")

    print("global cache")
    for k in GLOBAL_GRAPH_CACHE:
        print(k)

if __name__ == "__main__":
    main()