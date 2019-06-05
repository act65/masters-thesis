Related to;

- temporal abstraction
- deep versus wide representations in neural networks
- graphs


Define MDP. $\{S, A, r, T\}$

__A fully connected MDP.__
For every pair of states within the MDP, there is an action that allows the agent to transition between them. $\forall s_i, s_j \in S \exists a: s_j =\tau(s_i, a)$

If this is the case then the MDP reduces the the n-armed bandit problem.


__Using options to build FC-MDPs__

$$
\begin{align}
\omega &= a_0, a_1, \dots, a_k \tag{define an option}\\
T[\omega] &= \prod_{i=0}^k T[a_i] \\
s_{t+1} &= T[\omega]s_t \tag{a fully connected MDP}\\
\end{align}
$$

Here we have constructed a fully connected MDP from a MDP that isn't connected. The options

- Want to learn a set of actions / options that can easily traverse the state space.
- Less connectivity means the value of states are more "entangled"? (as to get to one state, you must go through others)


__Cts case__

Ok, let $S\in R^n, A \in R^m$. Want $\forall s_i, s_j \in S \exists a: s_j =\tau(s_i, a)$.

A necessary condition for this (assuming?) is that $\frac{\partial T}{\partial a}(s, a, s')$ is full rank for all $s, a, s'$.



***

- [HER](https://arxiv.org/abs/1707.01495)
- [Reachability](https://arxiv.org/abs/1707.01495)
- pick arbitrary states and set it as a goal
