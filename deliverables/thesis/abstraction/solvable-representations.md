Solving a problem via abstraction follows a generic formula. Transform the the problem into a new domain, solving the problem in this new domain, decode the solution back into the original domain.

In this section, we approach abstraction by identifying nice properties an optimisation problems might have and ways imbue an RL problems with those same properties.

A special case of this strategy has been employed in  mathematics and is known as a reduction. Where one type of problem is 'reduced' to another, ... SAT, P, NP, ...

And within RL, examples of this strategy are...?  [convex RL](https://bodono.github.io/thesis/bod_thesis.pdf)


Another way of saying the above, is that we want to learn a represetation of the RL problem that yields efficient optimisation with known solvers.

<!-- Want to demonstrate the problem being solved -->

But, which types of MDP are easily solved (and why)? And how can we map our problem into these easily solved instansiations?


Which types of MDP are easily solved?

- Discrete state space and linear transition fn?
- We know how to solve linear systems of equations with $O()$.
- And we know how to use these solutions to calculate the optimal, policy: generalised policy iteration.
- (Convex?)
- Well conditioned transitions!? (/ topology?)
- Tabular MDPs for small enough size ($n\le 200,000$ states) can be analytically solved.
- Linear systems can be solved with computational complexity $\mathcal O(n^3)$.
- Dense rewards.
- Linear transitions (LMDPs -

Note that it will not always be possible to find an efficient solution to an MDP.
Are some MDPs just fundammentally harder to solve than others?
Could mention no free lunch.

<!-- Want an example -->

## Definition

What is the problem we are solving here?

Find $\pi^{* }$ given $P, r$.

## A tabular representation

Learn a tabular MDP representation of the RL problem.

Why would we want to do this?
- Policy evaluation is expensive in the RL setting. The policy must be simulated over all possible states-action pairs. And scales poorly with variance. (how poorly?)
- ?

Just quickly, what does a tabular MDP look like?
- discrete states and actions
- $r$ and $P$ are simply look up functions, indexed by the current state-action.

$$
\begin{align}
V &= r_{\pi} + \gamma P_{\pi} V \tag{bellman eqn}\\
V - \gamma P_{\pi} V &= r_{\pi}\\
(I-\gamma P_{\pi})V &= r_{\pi}\\
V &= (I-\gamma P_{\pi})^{-1}r_{\pi}\\
\end{align}
$$

(finding the optimal policy is still a non-linear problem. how / why is it non-linear?!)


### Learning the (tabular) abstraction

Most recent advances in ML have been by finding clever ways to extend supervised learning techniques to unsupervised learning. Similarly, we can use supervised learning techniques, batch training, cross entropy, ... to train reward and transition approximations.

We are provided with examples $(s_t, a_t, r_t, s_{t+1}, a_{t+1})$. We can use these to...

$$
\begin{align}
\textbf  r \in \mathbb R^{n \times m}, &\; \textbf P \in [0,1]^{n \times m \times n} \\
L_{r} &= \text{min} \parallel r_t - \textbf r[\phi(s_t), a_t] \parallel^2_2 \tag{mean squared error}\\
L_{P} &= \mathop{\text{max}}_{\theta} \textbf P[\phi(s_{t+1}),\phi(s_t), a_t]\tag{max likelihood}\\
\end{align}
$$

### Policy iteration



### A linear representation

The bellman equation is a non-linear optimisation problem.
Is there a way to turn this into a linear problem? What do we need to sacrifice to do so?

$$
v(s) = \mathop{\text{max}}_a \Big[r(s, a) + \gamma \mathop{\mathbb E}_{s' \sim P(\cdot | s, a)} v(s') \Big]\\
$$

^^^ What makes it non-linear?!?


#### Linear markov decision problems (LMDPs)

How can we remove the sources of non-linearity from the bellman equation? The answer is a couple of 'tricks';

- rather than optimising in the space of actions, optimise in the space of possible transition functions.
- set the policy to be
- ?

Let's unpack these tricks and see how they can allow us to convert an MDP into a linear problem. And what the conversion costs.


$$
v(s) = \mathop{\text{max}}_u \Big[r(s) + \gamma \mathop{\mathbb E}_{s' \sim u(\cdot | s)} v(s') \Big] \\
\text{s.t.}  \;\; u(\cdot | s) \;\;\; ??? \\
$$

Could derive using Lagrangian multiplier?

$$
l(s, a) = q(s) + KL(u(\cdot | s) \parallel p(\cdot | s)) \\
v(x) = q(s) + \mathop{\text{max}}_a \Big[ KL(u(\cdot | s) \parallel p(\cdot | s)) +  \gamma \mathop{\mathbb E}_{x' \sim P(\cdot | x, a)} v(x') \Big]\\
$$

***

- How does computational complexity relate to sample complexity?
We are considering different problems. Sample complexity in MBRL comes from learning the transition and reward function. Computational complexity is considered (here) for the optimal control problem (only?).
- What about analytical solutions for cts problems?
