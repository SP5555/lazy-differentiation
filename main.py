import numpy as np
from auto_diff import *

array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

a = Variable(array1, "a")
b = Variable(array2, "b")
two = Variable(2.0)

expression: Multiply = (a + b) * two * b

# forward pass (Grade 1 math)
print(f"Forward: {expression.forward()}")

# derivative of expression with respect to "a"
print(f"dE/da: {expression.backward("a")}")

# derivative of expression with respect to "b"
print(f"dE/db: {expression.backward("b")}")