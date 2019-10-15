## Definition

Transfer is a type of generalisation of knowledge between domains. For this to be possible, the domains must share some 'similarity'.

Two environments, $t_i, t_j$ are similar if there exists an easy to find, structure preserving, mapping $f$ between them $f(t_i) = t_j$. While two environments might be isomorphic, depending on what we know about them it could be very hard to find a mapping from one to the other (at least polynomial time).

### Types of transfer?

Let $L$ be the test loss after training, and $T$ be the training task.
$$
\begin{align*}
L(T(B)) \le L(T(A) \to T(B)) \tag{Forward transfer}\\
L(T(A)) \le L(T(A) \to T(B)) \tag{Backward transfer} \\
\end{align*}
$$

So the key to this would be a decomposition of different types of knowledge?

$$
\begin{align*}
\dot L(T(B)) \le \dot L(T(A) \to T(B)) \\
\dot L(T(A)) \le \dot L(T(A) \to T(B)) \\
\end{align*}
$$

## MDPs

### Toy problems

Want to generate different MDPs that share various 'orders' of similarity.
If we mode each environment as a graph and the task is navigation, then it might be possible to easily generate graphs/rewards with structural similarities!? Various orders of persistent homology?

### Invariance

What do we mean by "transfer learning"? If we have two tasks/environments/action spaces/...?, $A, B$, then the performance of one task aids the other task.

A MDP is defined as

$$
M = \Big(S, \mathcal A, p(\cdot \mid s,a), R(s, a, s') \Big)
$$

- $S$: It is possible to change the state space, while preserving the dynamics. (??)
- $\mathcal A$: Change the action space, for example, instead of $\leftarrow, \rightarrow, \uparrow, \downarrow$ we use $\uparrow, \text{rot90}$
- $p(\cdot \mid s,a)$: from subtle things like not being able to reach a state on another one, to chan
- $R(s, a, s')$: A different reward funciton, aka a different task.

But one could imagine symmetries of $p(\cdot \mid s,a), R(s, a, s')$, such that some structure is preserved.

$$
\begin{align*}
p(\cdot \mid s,a) &:= T^{-1}(p(\cdot \mid T(s,a))) \\
&:= p(\cdot \mid T(s),a) \tag{equiv to transfer to a new state space}\\
&:= p(\cdot \mid s,T(a)) \\
R(s, a, s') &= T(R(s, a, s')) \\
\end{align*}
$$

### Analysis

What I would really like is a set of tools for analysing transfer learning. They should answer the following questions;

- what knowledge was transferred (high level, low level, ...?)
- how was it transferred? (if we are dealing with NNs then how does some knowledge get shared while other knowledge doesnt? because the existing knowledge allows faster learning?!)
- why was it transferred? (because the domains somehow shared similarities)

If we had a theory of transfer learning we would be able to;
- predict when X will transfer to Y.
- write down a pattern to generate representations for transfer between X/Y.
- __???__


***

What is necessary for future reseach on transfer learning.
