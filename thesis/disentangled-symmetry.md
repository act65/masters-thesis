## Intro: Symmetry.

> Want a representation that makes the problem easier to solve.

What is it?
Why do we care?
Invariants, equivariants. Quotients.

#### Conv nets

- Why are conv nets good: sample efficiency intuition.
- How could this be learned?

## Learning invaraiant representations

TODO find more refs on this.
- [Generalization Error of Invariant Classifiers](https://arxiv.org/abs/1610.04574)
- [On Learning Invariant Representation for Domain Adaptation](https://arxiv.org/abs/1901.09453)
- [?](?)

### Learning a measure of symmetry / similarity

Without a metric. Symmetry does make sense.
A symmetry is defined as the conservation of a ? under transformations.
We need a measure of that conserved quantity if we want to

For example; an apple classifier oracle. It tells us that a picture of an apple is still an apply if rotated, translated, sharpened, ...


__Completeness of the metric__

(_want to show that the way we build our repesentation is capable of capturing all symmetries_)

$$
\begin{align}
\text{Pick} \; (s, a, f, g) \; \text{s.t.}\; s' = f(a), a' = g_s(a) \\
???
\end{align}
$$


##### Invariant matching

Have a measure of invariance. Define an approximate similarity.

$$
\begin{align}
x \sim f(x) \;\text{ if }& \\
x &= f(x) \\
\chi(x) &\ge \epsilon : \quad \quad\chi(x) = x - f(x)  \\
\chi(x, x') &\ge \epsilon : \quad \quad\chi(x, x') = x - x'   \\
\end{align}
$$

Want to construct an function that matches this topology.
$h(x): \forall x \;$

$$
\chi(x) \ge \epsilon  \implies \parallel h(x) - h(f(x))\parallel \le \delta  \\
\chi(x, x') \ge \epsilon  \implies \parallel h(x) - h(x')\parallel \le \delta  \\
$$

But how to learn $h$ so it matches $\chi$s topology?


$$
\begin{align}
p_{\chi}(x, x') = \frac{e^{\chi(x, x')}}{Z} \\
L(h) = \mathop{\mathbb E}_{x, x'\sim p_{\chi}(x, x')}\parallel h(x) - h(x')\parallel \\
L(h) = -\mathop{\mathbb E} p_{\chi}(x, x') \log \frac{p_{h}(x, x')}{p_{\chi}(x, x')} \tag{KL}\\
\end{align}
$$

- Want a representation / function approximator, $h$, that makes calculating / sloving this efficient!

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

### Linear disentangled representations

If the linear disentangled representation. This implies a block diagonal structure on the gradients of the action of $G$. __!!!__
How can we flexibly parameterise a block diagonal strucutre?

And sparsity! NO it doesnt! That assumes statistical independence as well!?

For linear group representations, $g_i\in GL(\mathbb R^{n_i})$. And the action of $g_i$ on $x_i$ is matrix multiplication: $g_i \circ x_i = g_ix_i$. If we consider

By requiring some decomposition structure of $f,g$ we can learn the decomposition of the symmetry, and thus disentangle the representation?!
- Need a soft way to pick the number of dims.
- Need to ensure the diagonals are orthogonal (? maybe not ?).

$$F = \text{vstack}(F_{1:n}), \; f(x) = \text{kron}(\textbf 1(x) \cdot D)Fx$$
So we are designing some sort of gated attention mechanism. That selects the relevant transform(s).

## (linear disentangled) Representations for RL

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

## Generalisation and sample efficiency

Size of the initial space $n = \mid S \mid, m = \mid A \mid$.
Size of the quotient space $\tilde n = \mid S \mid, \tilde m = \mid A \mid$.
(need to derive the relationship between the two depending on the amount of symmetry)

Therefore, if we have inferred the structure of our MDP, then solving it requires $\mathcal O()$ samples, rathern than $\mathcal O(?)$

But what is the sample complexity of learning the symmetry?!?!!


## Related work

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


***

- where does compression come into disentanglement?!
-
