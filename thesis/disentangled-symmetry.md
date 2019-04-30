Invariance.
Groups.
Quotients.
Model reductions.


### Increased sample efficiency

?!?

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
