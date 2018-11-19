

[MoE](https://arxiv.org/abs/1701.06538)

Questions
- __Specialisation__! Independently useful contributions.
- Relationship to dropout!? [Information Dropout](https://arxiv.org/abs/1611.01353)
- __Q__ How is independence related to decomposition?
- Could use VQ-VAE somehow?
- Relationship to learning a basis!
- Relationship to invariance and symmetries!?

## Approaches

#### Structural decomposition

- Sparse gating, modules, ...

If we are using modular parts then we need to estimate gradients!

#### Implicit decomposition

- EWC, L-1, MI

If we are doing this implicitly, then ...? What geometry do we want!?

***

- Want to show a duality between these two views!!
- I would expect decomposition/disentanglement to be much easier in the RL setting as we get to take actions, to test independence!?



Other

[Discovering physical concepts](https://arxiv.org/abs/1807.10300)


## Basis

$$
\begin{align}
\forall x_i \in X \\
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
