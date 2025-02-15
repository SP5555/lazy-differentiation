# Auto Differentiation: Forward Accumulation

## Overview

**Forward Accumulation** is easier to understand mostly due to its similarity to how we (as humans, students, whatever) understand and calculate the derivatives by hand. It works by applying the chain-rule inside-out, (from inputs to outputs).

It calculates the gradients starting from the leaves (**Tensors**) of the expression tree and moving towards the root.

## How to use

1. Summon modules:

```python
import numpy as np
from auto_diff_forward import *
```

2. Define variables or **Tensors**:
    - Labels for **Tensors** are provided as the second parameter. These labels are used when differentiating with respect to that specific tensor. For constants, this parameter is optional.
```python
a = Tensor(np.array([1, 2, 3]), "a")
b = Tensor(np.array([4, 5, 6]), "b")
c = Tensor(np.array([7, 8, 9]), "c")
two = Tensor(2.0) # wrap inside Tensor if you wanna use a constant
```

3. Build the expression:

```python
expression = Log(b) * Tanh(Sqrt(a/b)) + a ** Sqrt(two * b)
```

4. Forward pass must be called first to situate the intermediate values in the expression tree graph. This is necessary for later `evaluate()` and `backward()` calls.

```python
expression.forward()
```

5. Then, enjoy the benefits:
    - `evaluate()` returns the value calculated by the forward pass.
    - `backward()` returns the gradient value with respect to the label passed into it.

```python
print(f"Forward: {expression.evaluate()}")
print(f"df/da  : {expression.backward('a')}")
print(f"df/db  : {expression.backward('b')}")
print(f"df/dc  : {expression.backward('c')}")
```

Printed Results:

```
Forward: [ 1.64063041  9.85328756 46.04784529]
df/da  : [ 3.10098914 14.32976344 52.04465632]
df/db  : [ 0.04738879  2.00435228 14.29276426]
df/dc  : [0. 0. 0.]
```