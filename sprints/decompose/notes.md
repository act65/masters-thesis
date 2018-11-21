Statistical view of disentanglement -> Kernel tests, ...?
Optimisation view of disentanglement -> ?

## Information bottleneck

> However, no algorithm is known to minimize the IB Lagrangian for non Gaussian, high-dimensional continuous random variables

hmph. someone should solve that...

When minimising $I(x, z)$, $z$ could be invariant to $x\sim p(x)$, but still be sensitive to other distributions?!?


## Approaches

#### Structural decomposition

- Sparse gating, modules, ...

If we are using modular parts then we need a way to learn the modules and to assign credit to each one!

#### Implicit decomposition

- EWC, L-1, MI

If we are doing this implicitly, then ...? What geometry do we want!? How can we recover a decomposition/??.


- Want to show a duality between these two views!!



Other

[Discovering physical concepts](https://arxiv.org/abs/1807.10300)


- Could use VQ-VAE somehow?
- Relationship to learning a basis!
- Relationship to invariance and symmetries!?
- Relationship to dropout!? [Information Dropout](https://arxiv.org/abs/1611.01353)

## Basis

$$
\begin{align}
\forall x_i \in X \\E[f(x)]
f^* , y^* = \mathop{\text{argmin}}_{f, y} D(x_i, f(y)) \tag{$y \in Y, f \in F$}\\
s.t. \mathop{\text{argmin}}_{Y, F} \mid Y\mid + \mid F \mid\\
\end{align}
$$

Imagine;
- $y_i$ is an ordered set of elements, say symmetries, $\sigma, \tau$, and $f$ is the composition of those elements.
-

What we want is to find a basis which we can efficiently use to construct observations.

Imagine;
- a set of images. We can search through combinations of the basis set to find a mixture that approximates $x_i$.


## Guassians -> max entropy

$$
\begin{align}
v &= E_{p(x)}[(x-\mu)^2] \\
&= \int p(x)(x-\mu)^2 dx \\
H(x) &= E[I(x)] \\
&= \int p(x) I(x) dx \\
&= \int -p(x)\ln(p(x)) dx \\
\frac{dH(x)}{dx} &=
\end{align}
$$
