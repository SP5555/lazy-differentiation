import sys
import os

# some issues
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from auto_diff_reverse import *
# from auto_diff_reverse.comp_node import GLOBAL_GRAPH_CACHE

def main():

    a = Tensor(np.array([1, 2, 3]))
    b = Tensor(np.array([4, 5, 6]))
    three = Tensor(3.0) # wrap inside Tensor if you wanna use a constant

    seed = np.array([1, 1, 1])

    expression = Sqrt(a ** b) / b

    np.set_printoptions(precision=8)

    expression.forward()
    expression.backward(seed=seed) # seed is gradient of loss

    print(f"Forward: {expression.evaluate()}")
    print(f"df/da  : {a.grad}")
    print(f"df/db  : {b.grad}")

if __name__ == "__main__":
    main()