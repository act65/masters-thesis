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

***

Given an MDP, $M_a = \{S, A, R, T\}$, construct another MDP from it as $M_b = \{S, A^k, R^k, T^k\}$. Where;
- an action in $M_b$ is the concatenation of $k$ actions in $M_a$.
- $T^k(s' | s, a) = T(T(\dots T(s, a_0), \dots), a_{k-1}), a_k)$

The computational complexity of solving the first MDP should be the same as the complexity of solving the second. $C(M_a) \equiv C(M_b)$.
We have traded action space complexity for state connectivity.

There are more actions, but they need to be taken less. Less problems with credit assignment.

***

Sparser is harder.
For any exploratory / non-deterministic policy there will be more variance in the value estimates when the transitions are sparse. (kind of)
We need to traverse through many different states to get to our goal. But our policy in those states might be not deterministic.
But in the fully connected case, there is little variance as we just pick the action / state we want to go to. Any variance is from the transition fn.

***

__Cts case__

Ok, let $S\in R^n, A \in R^m$. Want $\forall s_i, s_j \in S \exists a: s_j =\tau(s_i, a)$.

A necessary condition for this (assuming?) is that $\frac{\partial T}{\partial a}(s, a, s')$ is full rank for all $s, a, s'$.



***

- [HER](https://arxiv.org/abs/1707.01495)
- [Reachability](https://arxiv.org/abs/1707.01495)
- pick arbitrary states and set it as a goal
