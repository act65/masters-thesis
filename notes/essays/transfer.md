---
pagetitle: Notes on transfer learning
---

<div>


### Useful examples

!?!??!

### Related ideas

***

Two environments, $t_i, t_j$ are similar if there exists an easy to find, structure preserving, mapping $f$ between them $f(t_i) = t_j$. While two environments might be isomorphic, depending on what we know about them it could be very hard to find a mapping from one to the other (at least polynomial time).

- we need an approximate notion of equality, a distance or metric

$$
\begin{align}
f^* &= \mathop{\text{argmin}}_f d(f(t_i), t_j) \\
C(t_i, t_j) &= ??? \tag{the cost of finding $f^* $}\\
C(t_i, t_j) &\approx d(f^* (t_i), t_j) \tag{fixed computational budget}\\
\end{align}
$$

- What if $\text{min} \;d(f^* (t_i), t_j) >> 0$?
- ?

A representation $g: t \to h$ is a good one if, for many tasks, we can easily find similarities between any two tasks, $f_{ij}(g(t_i)) = g(t_j)$.

So a better representation would be easily able to map between more tasks/domains.

$$
\begin{align}
L &= E_{t_i \sim \mathcal T} E_{t_j \sim \mathcal T} [C(t_i, t_j)]\\
\end{align}
$$

Could apply this notion to the various $f_{ij}^{* }$ as well. How easily can they be transformed into each other?

- how much compute should be spent on trying to find similarities? when it could just be spent on learning the new domain. it depends on memory constraints...

## Graph similarity

Generate various graphs that share some known structure.
Want to be able to reason about symmetries in graphs or manifolds at many different scales and locations.

## Shared basis

Could imagine an over-complete basis, shared between task A and B?

Generate a basis, $\mathcal B$, maybe just a set of vectors.
Subsample and use to generate tasks. $\mathcal T_i \subset \mathcal B$.



## Composition

Have two goals. Want to compose them. Is it better to have policies or value fns or reward fns?

- How can you even compose policies? $\pi(s_t) = \pi_1(s_t) \cdot\pi_2(s_t)$.
- $V_1 \circ V_2 \neg \equiv R_1 \circ R_2$. It is possible for value fns to give weird results if compose incorrectly (when they depend on different policies!).

https://arxiv.org/pdf/1807.04439.pdf



## Transfer

Why do we want to do this? Transfer learning is the key to general intelligence!

### Definition

What do we mean by "transfer learning"? If we have two tasks/environments/action spaces/...?, $A, B$, then the performance of one task aids the other task.

A MDP is defined as
$$M = \Big(S, \mathcal A, p(\cdot \mid s,a), R(s, a, s') \Big)$$

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

For example, similarities between the reward in hockey and football. Get the round thing in the oppositions goal.

> Huh, never thought about it this way before. The states are an unordered set.
The transition fn provides all the structure on that space (much like an inner prod in Hilbert spaces?!?)
The neighbors of a state are the positions reachable from a single action.
No not quite. More like probabilistic vector maps? No that is only when combined with a policy.

Best current solutions!?

- successor representation/goal embeddings. $\to$ task transfer
- model-based RL (disentangle policy from model) allows transfer of control polices between environments and transfer of model between tasks in the same env.

Let $L$ be the test loss after training, and $T$ be the training task.
$$
\begin{align*}
L(T(B)) \le L(T(A) \to T(B)) \tag{Forward transfer}\\
L(T(A)) \le L(T(A) \to T(B)) \tag{Backward transfer} \\
\end{align*}
$$

Relationship to meta-learning. Different <i>'levels'</i> of knowledge can be transfered. In meta learning the low level details are not transferred, but the high level, "how to learn" lessons are transferred. So the key to this would be a decomposition of these different types of knowledge. __Q:__ How can these types of knowledge be disentangled!?

$$
\begin{align*}
\dot L(T(B)) \le \dot L(T(A) \to T(B)) \\
\dot L(T(A)) \le \dot L(T(A) \to T(B)) \\
\end{align*}
$$

### Analysis

What I would really like is a set of tools for analysing transfer learning.
I would like to be able to answer the questions;

- what knowledge was transferred (high level, low level, ...?)
- how was it transferred? (if we are dealing with NNs then how does some knowledge get shared while other knowledge doesnt?
because the existing knowledge allows faster learning?!)
- why was it transferred? (because the domains somehow shared similarities)

Seems quite related to representation learning. The key will be how knowledge is represented, and how easily that knowledge can be translated (/transformed)!?


If we had a theory of transfer learning we would be able to;
- predict when X will transfer to Y.
- write down a pattern to generate representations for transfer between X/Y.
- __???__

### Toy problems

Want to generate different MDPs that share various 'orders' of similarity.
If we mode each environment as a graph and the task is navigation, then it might be possible to easily generate graphs/rewards with structural similarities!? Various orders of persistent homology?


</div>
