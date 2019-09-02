## Related ideas

- gradient coupling
- weight sharing
- data sharing
- data augmentation

## Why do we care?

Efficient learning.
- Variance reduction. Fewer samples required. Faster convergence.
- Control generalisation.

### Variance reduction

(lots of people trying to do this. control variantes... etc)
[Invariance reduces Variance: Understanding Data Augmentation in Deep Learning and Beyond](https://arxiv.org/abs/1907.10905)

> How do less variant gradient estimates lead to faster convergence?



### Controlling generalisation

> How should a NN generalise?

$$
H = \langle \nabla_{\theta} f_{\theta}(x), \nabla_{\theta} f_{\theta}(x)\rangle \\
\dot f(x) =  -\eta H(x) \nabla_{f} \mathcal L(x) \\
$$

Generalise the information gained at $j$ other points to update $x_i$. The updates are weighted by how similar their gradients are.

Why would we want to do that?

- If many other samples are pointing in the same direction, take a larger step. Acceleration! (how much of this is a result of averaging the gradients together before updating?) This is kind of trivial...?
-

Imagine we have some (as of yet) unobserved data point, $x_i$. We are interested in what this data point is likely to be classified as, when we finally see it. What it is classified as is dependent on its similarity to the data points we have seen, $\{x_j\}$.
$$
\dot f(x_i) =  -\eta\sum_j \langle \nabla_{\theta} f_{\theta}(x_i), \nabla_{\theta} f_{\theta}(x_j)\rangle \cdot \nabla_{f} \mathcal L(x_j) \\
f(x_i, \tau) = f(x_i){_0} + \int_0^\tau \dot f(x_i) dt
$$
Assuming $y(x_i)$ was initialised at ???, then we can track the dynamics of

$$

$$

> How should these $\{x_j\}$ be used to update $f$ so that we will accurately classify $x_i$ when we see it?



##### Other Kernels

- $k(x_i, x_j) = \langle x_i, x_j\rangle$. If their inputs are similar then their outputs should share gradients (aka their outputs should also be similar).
- $k(x_i, x_j) = \langle x_i, x_j\rangle$
