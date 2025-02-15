# LAZY Differentiation

**L**ow-effort **A**utomatic **Z**eroing **Y**our-workload Differentiation module for backpropagation in neural networks.

Forward Accumulation test code: [auto_diff_forward/dev/main.py](auto_diff_forward/dev/main.py)

Reverse Accumulation test code: [auto_diff_reverse/dev/main.py](auto_diff_reverse/dev/main.py)

## First of all, what is the problem?

Let's say you've got this absolute monstrosity (but trust me, you will never, ever need this in your entire life, not even in the next one).

```math
f(a,b) =\ln (b)\tanh\left(\sqrt{\frac{a}{b}}\right)+a^{\sqrt{2b}}
```

And now, for some reason, you want the partial derivative with respect to $a$.

```math
\frac{\partial f}{\partial a} =\frac{\ln( b){sech}^{2}\left(\sqrt{\frac{a}{b}}\right)}{2b\sqrt{\frac{a}{b}}} +\sqrt{2b}\left( a^{\sqrt{2b} -1}\right)
```

While you're at it, why not grab the partial derivative with respect to $b$ too.

```math
\frac{\partial f}{\partial b} =-\frac{a\ln( b){sech}^{2}\left(\sqrt{\frac{a}{b}}\right)}{2b^{2}\sqrt{\frac{a}{b}}} +\frac{a^{\sqrt{2b}}\ln( a)}{\sqrt{2b}} +\frac{\tanh\left(\sqrt{\frac{a}{b}}\right)}{b}
```

You just witnessed an intellectual nuclear detonation. I had to use a derivative calculator. Fair enough.

You can choose to hardcode this entire mess into your code... but committing Seppuku might be an easier option at this point.

## So, how does **Automatic Differentiation** save our lives?

*Note: This example uses `auto_diff_forward` module (Forward Accumulation mode).*

Summon the modules:

```python
import numpy as np
from auto_diff_forward import *
```

Define what we need:

```python
    a = Tensor(np.array([1, 2, 3]), "a")
    b = Tensor(np.array([4, 5, 6]), "b")
    two = Tensor(2.0) # wrap inside Tensor if you wanna use a constant
```

Construct the expression. Throw everything together like this:

```python
    expression = Log(b) * Tanh(Sqrt(a/b)) + a ** Sqrt(two * b)
```

Then compute, print them out and see the magic:

```python
    # forward to situate the tensors in the expression tree
    expression.forward()

    np.set_printoptions(precision=8)

    # forward pass (Grade 1 math)
    print(f"Forward: {expression.evaluate()}")

    # Easy Lazy Derivatives

    # derivative of expression with respect to "a"
    print(f"df/da  : {expression.backward('a')}")

    # derivative of expression with respect to "b"
    print(f"df/db  : {expression.backward('b')}")
    
    # derivative of expression with respect to "c"
    # (which doesn’t exist, so should be all zeroes)
    print(f"df/dc  : {expression.backward('c')}")
```

Check the results. It prints:

```
Forward: [ 1.64063041  9.85328756 46.04784529]
df/da  : [ 3.10098914 14.32976344 52.04465632]
df/db  : [ 0.04738879  2.00435228 14.29276426]
df/dc  : [0. 0. 0.]
```

Compare them with calculator-computed values (*Desmos to the rescue!*).

| | $a=1,b=4$ | $a=2,b=5$ | $a=3,b=6$ |
| :---: | :---: | :---: | :---: |
| $f(a,b)$ | $1.64063040929$ | $9.85328756026$ | $46.0478452926$ |
| $\frac{\partial f}{\partial a}$ | $3.10098913913$ | $14.3297634449$ | $52.0446563188$ |
| $\frac{\partial f}{\partial b}$ | $0.0473887857196$ | $2.00435228287$ | $14.2927642565$ |
| $\frac{\partial f}{\partial c}$ | $0.0$ | $0.0$ | $0.0$ |

AWESOME, right?! Yeah! :fire:

Under the hood, it is just an *absurd* amount of Chain Rule doing their own things. There is no magic. Sadge.

No manual calculus, no more headaches. **Welcome to Automatic Differentiation!**

# Modules

Oops, these `README.md`s only have how-to-use in it. Hopefully, more details are added tomorrow. ~~Tomorrow never comes.~~

Forward Accumulation Mode: [auto_diff_forward/README.md](auto_diff_forward/README.md)

Reverse Accumulation Mode: [auto_diff_reverse/README.md](auto_diff_reverse/README.md)