Invariance.
Groups.
Quotients.
Model reductions.


### Increased sample efficiency

- Want to generalise to new pairs! $\langle s,a\rangle, \langle s', a'\rangle$. (zero shot)
- Want to generalise past experience! After figuring out two states are similar, share their data. (one shot)

### Completeness

(_want to show that the way we build our repesentation is capable of capturing all symmetries_)

$$
\begin{align}
\text{Pick} \; (s, a, f, g) \; \text{s.t.}\; s' = f(a), a' = g_s(a) \\
???
\end{align}
$$



##### Symmetry learning for function approximation in RL
(https://arxiv.org/abs/1706.02999)

They use reward trajectories to construct a notion of similarity between two state-action pairs.

This technique allows one-shot generalisation to new state-action pairs. A new state-action pair is observed, and it has the sample reward trajectory as another state-action pair. We can automatically transfer the value estimate from the former pair to the latter. (for the current policy...).
__Actually. No?!!?__

Depending on how much symmetry is displayed, we are increasing the amount of data each state - action pair has...? How much more data per symmetry?
A form a weight sharing!
This is thought to be why conv nets work so well, their kernels recieve huge amounts of data as they are shared over many spatial locations.

Problems.
- It requires discrete state / action spaces, as to represent the ??? they use a tree.
- It doesnt scale well. Number of possible future trajectories = ???.
- Requires dense reward

$N: [n_{states}\times n_{actions} \times l_{steps} \times k_{possibilities}]$

$$
\begin{align}
N(s, a, l, k) &= ??? \\
\sum_{l, k} &= \sum_{l=l_o}^L \sum_{k_o}^K \\
\chi(\langle s,a\rangle, \langle s', a'\rangle) &= \frac{\sum_{l, k} \text{min} (N(s, a, l, k), N(s', a', l, k))}{\sum_{l, k} N(s, a, l, k) \sum_{l, k} N(s', a', l, k))} \\
\chi(\langle s,a\rangle, \langle s', a'\rangle) &= \frac{\sum_{l, k} p(\tau^k_l | s, a) p(\tau^k_l | s', a')}{\sum_{l, k} p(\tau^k_l | s, a) \sum_{l, k} p(\tau^k_l | s', a'))} \\
\chi(\langle s,a\rangle, \langle s', a'\rangle) &= JS(p(\cdot| s, a) \parallel p(\cdot | s', a'))\\
\end{align}
$$

^^^ This reminds me of testing for statistical independence!? $1 = \frac{P(A,B)}{P(A) P(B)}$

The estimates of $D(p(\cdot | s, a), p(\cdot | s', a'))$ are independent of policy. Sure, the current policy will effect the distribution of trajectories $p(\cdot | s, a)$. But if $\exists f, g: s' = f(s), a' = g_s(a)$ then $p(\cdot | s', a')$ will be effected equally.
But, if $p(\cdot | s, a)$ changes allot, then we need to explore enough to also update $p(\cdot | s', a')$. Hmm. Would rather couple the two?! $p(\cdot | s, a) = p(\cdot | s', a')$ once we have figured out that they are 'similar' (under a stationary policy). Want to generalise to other policies.

$$
 \chi(\langle s,a\rangle, \langle s', a'\rangle) >\Delta \\ \implies \mathop{\text{min}}_{\theta} D\big( \zeta(s, a), \zeta('s, a')\big) \\
\mathop{\text{min}}_{\theta} \mathop{\mathbb E}_{\chi} \big[\parallel \zeta(s, a) - \zeta(s', a') \parallel_2^2 \big]\\
$$

- Problem. We have gained data efficiency, but not computational effeciency? We need to train the network for each of these symmetries.
- Question. If we are training a NN in this way, how does the invariance get implemented within the NN?
- As training proceeds, and more symmetries have been observed. There might be very many pairs that are 'similar'. Want to visualise these clusters?!
- Oh... All we are doing is clustering based on a similarity measure... How does that relate to symmetry and quotients?


> Want an abstraction that makes the problem easier to solve.

How does disentangled states and / or actions make it easier to solve?
???


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

***

If the linear disentangled representation. This implies a block diagonal structure on the gradients of the action of $G$. __!!!__
How can we flexibly parameterise a block diagonal strucutre?

And sparsity! NO it doesnt! That assumes statistical independence as well!?

For linear group representations, $g_i\in GL(\mathbb R^{n_i})$. And the action of $g_i$ on $x_i$ is matrix multiplication: $g_i \circ x_i = g_ix_i$. If we consider


__HOW?__ Learning group decompositions

Learn a similarity measure $\chi(\langle s,a\rangle, \langle s', a'\rangle)$ based on data (maybe reward trajectories).


A pair is similar if we can find a transformation that $\chi$ is invariant to.
Given $f,g$ we can learn $\chi$. Or vice versa.

$$
\begin{align}
L(f, g) &= \mathop{\mathbb E}_{\langle s,a\rangle, \langle s', a'\rangle \sim \chi} \parallel \phi(s) - f(\phi(s')) \parallel + \parallel \varphi(a) + g_s(\varphi(a')) \parallel \\
\end{align}
$$

Given a similarity measure. Want to find its invariant transformations!

By requiring some decomposition structure of $f,g$ we can learn the decomposition of the symmetry, and thus disentangle the representation?!
- Need a soft way to pick the number of dims.
- Need to ensure the diagonals are orthogonal (? maybe not ?).

$$F = \text{vstack}(F_{1:n}), \; f(x) = \text{kron}(\textbf 1(x) \cdot D)Fx$$
So we are designing some sort of gated attention mechanism. That selects the relevant transform(s).

$$
\begin{align}
\parallel \phi(s) - f(\phi(s')) \parallel \tag{equvariance of $\phi$ to $f$} \\
\parallel \phi(s) - \phi(f(s')) \parallel \tag{invariance of $\phi$ to $f$} \\
\end{align}
$$


***


Have a function $f(x)$ (where $x = (y, z)$?)
Want to find its invariants, $g_i\in G$.

$$
\begin{align}
g_i(f(x)) &= f(g_i(x)) \tag{equivariance} \\
L(g_i) &=  \mathop{\mathbb E}_xD(g_i(f(x)), f(g_i(x))) \\
&=  \mathop{\mathbb E}_{x, x'=T(x)} D(f(x), g_i^{-1}(f(x')) \\
L &= \sum_{i\neq j} \langle g_i, g_j \rangle + \sum_i L(g_i) \\
L &=   \tag{optimise a group}\\
\end{align}
$$

How to optimise for a group? Closure, inverse, associativity, id, ...?

equivariance. aka commutativity. aka homomorphism. aka ...?
