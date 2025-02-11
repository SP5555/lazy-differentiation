import sys
import os

# some issues
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from auto_diff import *

def main():
    array1 = np.array([1, 2, 3])
    array2 = np.array([4, 5, 6])

    a = Tensor(array1, "a")
    b = Tensor(array2, "b")
    two = Tensor(2.0)

    expression = (a + b) * (a / b)

    # forward pass (Grade 1 math)
    print(f"Forward: {expression.forward()}")

    # derivative of expression with respect to "a"
    print(f"dE/da: {expression.backward('a')}")

    # derivative of expression with respect to "b"
    print(f"dE/db: {expression.backward('b')}")

if __name__ == "__main__":
    main()