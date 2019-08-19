### Towards a definition of disentangled representations
(https://arxiv.org/abs/1812.02230)

Linear disentangled representations.

We know that a set of obsevations have group structure $G=G_1\times \dots G_n$ (how do we know that? dont ask).
We say that the representation (of the observations) $x \in X$ is disentangled if

$$
\begin{align}
&\forall g\in G, \forall x \in X: \\
g\circ x &= (g_1, \dots, g_n) \circ (x_1 \dots x_n)\\
& =(g_1 \circ_1 x_1, \dots g_n \circ_n x_n) \\
\end{align}
$$

##### Linear disentangled representations

If the linear disentangled representation. This implies a block diagonal structure on the gradients of the action of $G$. __!!!__
How can we flexibly parameterise a block diagonal strucutre?

And sparsity! NO it doesnt! That assumes statistical independence as well!?

For linear group representations, $g_i\in GL(\mathbb R^{n_i})$. And the action of $g_i$ on $x_i$ is matrix multiplication: $g_i \circ x_i = g_ix_i$. If we consider

By requiring some decomposition structure of $f,g$ we can learn the decomposition of the symmetry, and thus disentangle the representation?!
- Need a soft way to pick the number of dims.
- Need to ensure the diagonals are orthogonal (? maybe not ?).

$$F = \text{vstack}(F_{1:n}), \; f(x) = \text{kron}(\textbf 1(x) \cdot D)Fx$$
So we are designing some sort of gated attention mechanism. That selects the relevant transform(s).

##### (linear disentangled) Representations for RL

A pair is similar if we can find a transformation that $\chi$ is invariant to.
Given $f,g$ we can learn $\chi$. Or vice versa.

$$
\begin{align}
L_{\chi} &= \langle s,a\rangle, \langle s', a'\rangle \\
L_{(f, g)} &= ? \\
L_{\phi} &= \mathop{\mathbb E}_{\langle s,a\rangle, \langle s', a'\rangle \sim \chi} \sum_{f\in F}\parallel \phi(s) - f(\phi(s')) \parallel + \sum_{g\in G}\parallel \varphi(a) + g_s(\varphi(a')) \parallel \\
\end{align}
$$

Given a similarity measure. Want to find its invariant transformations!



## Learning equivariant representations

> aka disentanglement

Typical setting.
Pick $z$ from $Z$. This is our generating code. The true factors within the data we see. We assume $Z$ is disentangled.
Now, $x\in X$ are created by $f: Z \to X$.

Alternatively we can think about disentanglement as invariance to transformation. $f(g(x)) = f(x)$. A nicer property is equivariance, $f(g \circ x)) = g \circ f(x))$. Or maybe $f(g_1 \circ \dots g_n \circ x) = g \circ \dots g_n \circ f(x))$.

We could reframe 1) as $z = z_1 \circ \dots z_n = z_1 + \dots z_n$. Giving, $x = f(z_1 \circ \dots z_n)$.

So without a metric, what can be done? Obviously symmetries dont make sense, unless has happen to have some priors on a useful metric for the task. Thus we must do something like ICA / PCA?!

Why do we care?!?

- Want to generalise to new pairs! $\langle s,a\rangle, \langle s', a'\rangle$. (zero shot)
- Want to generalise past experience! After figuring out two states are similar, share their data. (one shot)
- BUT specifically to do with equivariance!?!? better generalisation!?

### Learning disentangled representations
> Invariance -> Equivariance.

1. Find invariant transformations, $g$, under $M(x)$
1. Learn a representation $\phi$ that has equivariance to those transforms  $g\circ\phi(x) = \phi(g\circ x)$

```python
# Given M(x)
G = find_invariants(M)  # the set of transforms M is invariant to
f = construct_representation(G)
```

##### Transform recovery

Have an invariant measure, but want use it to build an equivariant representation. Aka, want to find the invariant transforms, then use to build the representation!?

$$
\begin{align}
\parallel \phi(s) - \phi(f(s')) \parallel \tag{invariance of $\phi$ to $f$} \\
\parallel \phi(s) - f(\phi(s')) \parallel \tag{equvariance of $\phi$ to $f$} \\
\end{align}
$$

Have a function $f(x)$ (where $x = (y, z)$?)
Want to find its invariants, $g_i\in G$.

$$
\begin{align}
g_i(f(x)) &= f(g_i(x)) \tag{equivariance} \\
L(g_i) &=  \mathop{\mathbb E}_xD(g_i(f(x)), f(g_i(x))) \\
&=  \mathop{\mathbb E}_{x, x'=T(x)} D(f(x), g_i^{-1}(f(x')) \\
\end{align}
$$


## Learning group decompositions

Group discovery. Want to infer group structure examples.
Two levels?! Subgroups and their symmetries.

equivariance. aka commutativity. aka homomorphism. aka ...?

##### Group completion

Given the binary relation and some elements of the group. Solve for the rest.

How many elements do you need? Does it depend on the relation??

Can be done in many simple cases for example. ${+}_{mod 5}, \{0,1,3,4\}$.


How to optimise for a group? Closure, inverse, associativity, id, ...?

Optimise for group structure.

$$
\begin{align}
L_A &= \tag{associativity} \\
L_C &=  ??? \tag{closure}\\
L_{id} &= \tag{identity} \\
L_{inv} &=  ??? \tag{inverse}\\
\end{align}
$$
