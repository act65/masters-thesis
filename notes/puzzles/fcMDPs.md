An MDP is normally defined as the set $\{S, A, T, r\}$. But consider a 'fully connected MDP' (fcMDP). Defined as an MDP with the extra requirement that: for any pair of states within the MDP, $A, B$, there is an action that allows the agent to transition from $A$ to $B$. That is, $\forall s_i, s_j \in S\;\;\exists a\in A: \tau(s_j | s_i, a)\approx 1$.

Intuitively, this makes the problem a lot easier. Rather than having to navigate through a maze, we simply pick the action that takes us to the exit. But, on closer inspection, it isn't that simple...

(problem here? probability = 1?!? unsure)

Why would we want to do this?

> __Lemma__ A fcMDP can be solved in ???.

Proof?!?

> __Lemma__ A fully connected MDP is just a n-armed bandit problem.

Proof?!?

So fcMDPs have some nice properties, but how does this help us solve MDPs? Consider how we might transform a MDP into a fcMDP.

What needs to happen to a MDP for it to become a fcMDP? We need to transform the action space so that it better connect the states.

Firstly, we are going to need more actions. For each state to be reachable by ... we need $|A| = |S|$.

$$
\begin{align}
\omega &= a_0, a_1, \dots, a_k \tag{define an option}\\
T[\omega] &= \prod_{i=0}^k T[a_i] \\
s_{t+1} &= T[\omega]s_t \tag{a fully connected MDP}\\
\end{align}
$$

Here we have constructed a fully connected MDP from a MDP that isn't connected. This required a change of action space, from the ground actions to a set of temporally abstract options.

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




## Contextual decision problems

Desicions are made in stages.

But what if they were not?! Rather than adapting decision to new observations, you picked actions before you knew

- Decision problem $|A|^H$
- Contextual decision problem $|S\times A|^H$.

If adaption decisions based on history us always advantaegous, then decision problems should give an upper boun on the performance of a sequential learner.

__Q__ Is there an intermediate setting between n-armed bandits and sequential learners that controls how much information (about the past) a learner can use?



***

- Relationship to autoregressive models?
- Delay a choice for as long as possible??? Then you will have more information.


Related to;

- temporal abstraction
- deep versus wide representations in neural networks
- graphs


### Actions $\to$ Options. And back. Transform


$$
\mathcal M = \{S, A, r, P, \gamma\} \\
M_1, M_2 \in \mathcal M \\
T_k: \mathcal M \to \mathcal M \\
\\
\{S, A, r, P\} \to \{S, A^k, \hat T_k(r), \hat T_k(P)\} \\
\hat T_k(r) = \sum_{i=t}^{t+k} r(s_i, a_i) \\
\hat T_k(P) = \prod_{i=t}^{t+k} P(s_{i+1} | s_i, a_i) \\
\\
T_k^{-1}: \mathcal M \to \mathcal M \\
\{S, A^k, \hat T_k(r), \hat T_k(P)\} \to  \{S, A, r, P\} \\
$$

Is this transform;
- invertible?
- homomorphic (under which ops?)
- isomorphic

But. Why would we care?

- less variance
- actions having higher correlation with rewards
-

$$
\text{var}(V_M^\pi(s)) \le \text{var}(V_{\hat M}^\pi(s)) \\
\mathop{\mathbb E}_{R\sim M(s, \pi)}[(R-V_M^\pi(s))^2] \le \mathop{\mathbb E}_{R\sim \hat M(s, \pi)}[(R-V_{\hat M}^\pi(s))^2] 
$$
