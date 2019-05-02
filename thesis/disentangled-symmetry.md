Invariance.
Groups.
Quotients.
Model reductions.


### Increased sample efficiency

Size of the initial space $n = \mid S \mid, m = \mid A \mid$.
Size of the quotient space $\tilde n = \mid S \mid, \tilde m = \mid A \mid$.

(need to derive the relationship between the two depending on the amount of symmetry)

Therefore, if we have inferred the structure of our MDP, then solving it requires $\mathcal O()$ samples, rathern than $\mathcal O(?)$

But what is the sample complexity of learning the symmetry?!?!!

### Group theory (plus ML) puzzles

Group discovery. Want to infer group structure examples.
Two levels?! Subgroups and their symmetries. 

##### Group completion

Given the binary relation and some elements of the group. Solve for the rest.

How many elements do you need? Does it depend on the relation??

Can be done in many simple cases for example. $+_{mod 5}, \{0,1,3,4\}$.


##### Invariant transform recovery


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


### Disentanglement

Typical setting.
Pick $z$ from $Z$. This is our generating code. The true factors within the data we see. We assume $Z$ is disentangled.
Now, $x\in X$ are created by $f: Z \to X$.

Alternatively we can think about disentanglement as invariance to transformation. $f(g(x)) = f(x)$. A nicer property is equivariance, $f(g \circ x)) = g \circ f(x))$. Or maybe $f(g_1 \circ \dots g_n \circ x) = g \circ \dots g_n \circ f(x))$.

We could reframe 1) as $z = z_1 \circ \ dots z_n = z_1 + \ dots z_n$. Giving, $x = f(z_1 \circ \ dots z_n)$.
