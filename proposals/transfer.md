% transfer
# Transfer

Why do we want to do this? Transfer learning is the key to general intelligence!

## Definition

What do we mean by transfer in "transfer learning"? If we have two tasks/environments/action spaces/...?, $A, B$, then the performance of one task aids the other task.

A MDP is defined as
$$M = \Big(S, \mathcal A, p(\cdot \mid s,a), R(s, a, s') \Big)$$

- $S$: It is possible to change the state space, while preserving the dynamics. (??)
- $\mathcal A$: Change the action space, for example, instead of $\leftarrow, \rightarrow, \uparrow, \downarrow$ we use $\uparrow, \text{rot90}$
- $p(\cdot \mid s,a)$: from subtle things like not being able to reach a state on another one, to chan
- $R(s, a, s')$: A different reward funciton, aka a different task.

But one could imagine symmetries of $p(\cdot \mid s,a), R(s, a, s')$, such that some structure is preserved.

$$
\begin{align*}
p(\cdot \mid s,a) &= T^{-1}(p(\cdot \mid T(s,a))) \\
&= p(\cdot \mid T(s),a) \tag{equiv to transfer to a new state space}\\
&= p(\cdot \mid s,T(a)) \\
R(s, a, s') &= T(R(s, a, s')) \\
\end{align*}
$$

For example, similarities between the reward in hockey and football. Get the round thing in the oppositions goal.

<p>Huh, never thought about it this way before. The states are an unordered set.
The transition fn provides all the structure on that space (much like an inner prod in Hilbert spaces?!?)
The neighbors of a state are the positions reachable from a single action.
No not quite. More like probabilistic vector maps? No that is only when combined with a policy.</p>

Best current solutions!?

- successor representation/goal embeddings. $\to$ task transfer
- model-based RL (disentangle policy from model) allows transfer of control polices between environments and transfer of model between tasks in the same env.
- why was it transferred? (because the domains somehow shared similarities)

$$
\begin{align*}
L(T(B)) \le L(T(A) \to T(B)) \tag{Forward transfer}\\
L(T(A)) \le L(T(A) \to T(B)) \tag{Backward transfer} \\
\end{align*}
$$

Relationship to meta-learning. Different <i>'levels'</i> of knowledge can be transfered.

- in transfer learning we tend to deal with absolute knowledge about one domain (say a robot simulation)
- meta learning transfers <i>meta</i> knowledge about how to learn (thus it is also called learning to learn)

How can we tease apart these definitions into a heirarchy?

$$
\begin{align*}
\dot L(T(B)) \le \dot L(T(A) \to T(B)) \\
\dot L(T(A)) \le \dot L(T(A) \to T(B)) \\
\end{align*}
$$

## Analysis

What I would really like is a set of tools for analysing transfer learning.
I would like to be able to answer the questions;


- what knowledge was transferred (high level, low level, ...?)
- how was it transferred? (if we are dealing with NNs then how does some knowledge get shared while other knowledge doesnt?
because the existing knowledge allows faster learning?!)
- why was it transferred? (because the domains somehow shared similarities)

Seems quite related to representation learning. The key will be how knowledge is represented, and how easily that knowledge can be translated (/transformed)!?

Like a communication channel. I want to see what has been communicated!
Attributing performance on a current task to a past experiences/tasks. (good at snowboarding bc i used to surf)


If we had a theory of transfer learning we would be able to;
- predict when X will transfer to Y.
- write down a pattern to generate representations for transfer between X/Y.
- ?

## Toy problems

Two environments, only difference is visual appearance, ...
Two environments,

Want to generate different MDPs that share various 'orders' of similarity.
Navigation on graphs? Various orders of persistent homology?

## Heirarhcal RL

__Conjecture__: meta/transfer/continual/... learning naturally emerge from heirarchical/multi-scale RL.
We naturally see a decomposition of the tasks into what they share at different levels of abstraction (not just time!?).

__TODO__: Heirarchical filters and meta RL and options.

$$
\begin{align*}
z_t &= f(s_t, a_t, r_t^k) \\
\pi(s_t) &= g(\sum_k f_k(z_t)) \\
v_t^k &= h_k(z_t) \\
\mathcal L_k &= \sum \parallel v_t^k - R(\gamma_k) \parallel \\
R(\gamma) &= \sum \gamma^i r_i \\
\end{align*}
$$

Ahh. But need to be heirarchically propagating forward differences __!!!__ ? Else would need very large compute to fit in large enough batches...

## Resources

- [Learning to learn](https://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/)
- [Elastic weights consolidation](https://arxiv.org/abs/1612.00796)
- [Successor features](https://arxiv.org/abs/1606.05312)
