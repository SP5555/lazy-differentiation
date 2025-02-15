# Auto Differentiation: Reverse Accumulation

## Overview

**Reverse Accumulation** might feel a bit more abstract, not immediately intuitive, but it’s incredibly efficient, especially for complicated functions (*neural networks*). It works by applying the chain-rule outside-in (from outputs to inputs).

In this mode, gradients are computed starting from the root of the expression tree and propagate backward toward the leaves (**Tensors**). This process accumulates the gradients in each node as we move through the tree.

## How to use

1. Summon modules:

```python
import numpy as np
from auto_diff_reverse import *
```

2. Define variables or **Tensors**:
    - Unlike **Forward Accumulation**, labels are not needed. Gradients are automatically accumulated in their respective base **Tensors** without the need to specify a label.
```python
a = Tensor(np.array([1, 2, 3]))
b = Tensor(np.array([4, 5, 6]))
c = Tensor(np.array([7, 8, 9]))
two = Tensor(2.0) # wrap inside Tensor if you wanna use a constant
```

3. Build the expression:

```python
expression = Log(b) * Tanh(Sqrt(a/b)) + a ** Sqrt(two * b)
```

4. Forward pass must be called first to situate intermediate values in the expression tree graph, followed by the backward pass call to propagate backwards and calculate gradients for all nodes in the tree. This is necessary for later `evaluate()` and `grad` calls.

```python
expression.forward()
expression.backward(seed=np.array([1, 1, 1]))
```

What’s the deal with the SEED? In simple terms, if you just want the "normal" derivative, pass in `1.0` or a numpy array of ones with the same dimensions. But here's the kicker: the seed plays a crucial role in backpropagation for neural networks, as it directly represents the gradient of the loss in the expression.

5. Then, enjoy the benefits:
    - `evaluate()` returns the value calculated by the forward pass.
    - `grad` is accessed on **Tensors** because, by this point, the backward pass has already populated them with the gradient values.

```python
print(f"Forward: {expression.evaluate()}")
print(f"df/da  : {a.grad}")
print(f"df/db  : {b.grad}")
print(f"df/dc  : {c.grad}")
```

Printed Results:

```
Forward: [ 1.64063041  9.85328756 46.04784529]
df/da  : [ 3.10098914 14.32976344 52.04465632]
df/db  : [ 0.04738879  2.00435228 14.29276426]
df/dc  : [0. 0. 0.]
```

6. After the backward pass, `zero_grad()` must be called to reset the accumulated gradients to zero if we plan to do another backward pass with different seed.

```python
a.zero_grad()
b.zero_grad()
c.zero_grad()
```