> Want an abstraction that makes the problem easier to solve.

What about an abstraction that reduces the problem to a discrete state space and linear transition fn? That seems nice.
- We know how to solve linear systems of equations with $O()$.
- And we know how to use these solutions to calculate the optimal, policy: generalised policy iteration.

Efficient incremental estimation of the optimal policy.
This is the real goal!?

## A linear representation and policy iteration

Learn a tabular MDP representation of the Rl problem.

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


## Learning the (linear) abstraction

Most recent advances in ML have been by finding clever ways to extend supervised learning techniques to unsupervised learning. Similarly, we can use supervised learning techniques, batch training, cross entropy, ... to train reward and transition approximations.

We are provided with examples $(s_t, a_t, r_t, s_{t+1}, a_{t+1})$. We can use these to...

$$
\begin{align}
\textbf  r \in \mathbb R^{n \times m}, &\; \textbf P \in [0,1]^{n \times m \times n} \\
L_{r} &= \text{min} \parallel r_t - \textbf r[\phi(s_t), a_t] \parallel^2_2 \tag{mean squared error}\\
L_{P} &= \mathop{\text{max}}_{\theta} \textbf P[\phi(s_{t+1}),\phi(s_t), a_t]\tag{max likelihood}\\
\end{align}
$$

## Evaluating the learned abstraction

But what about the error in our approximation?
Error in the reward doesnt seem like such a big deal? (why?!?)

- Is the approximation error, $E$ likely to be biased in any way? Certain states or actions having more error than others? Or can we just model it as noise?
- Are some of the elements likely to be correlated? Or can we sample the noise IID?  
- If so, what sort of noise is $E$. Want uniform noise on a simplex. One draw for each state.

$$
\begin{align}
E &\sim \mathcal N, \parallel E \parallel_{\infty} < \delta \\
\hat P &= P + E \\
\hat V &= (I-\gamma \hat P \pi)^{-1}r \pi \\
\epsilon &= |V - \hat V |\\
&= |(I-\gamma P \pi)^{-1}r \pi -  (I-\gamma (P + E) \pi)^{-1}r \pi| \\
&= |\Big((I-\gamma P \pi)^{-1} -  (I-\gamma P\pi + \gamma E\pi)^{-1} \Big)r \pi| \\
\end{align}
$$

Want to find $X$ such that $(I-\gamma P\pi)^{-1} - (I-\gamma P\pi + \gamma\Delta\pi)^{-1} = X$. or an upper bound on $X$?

Hmph.
- Why are we inverting.
- What does the inverse do? How does it deal with small pertubations?
- https://en.wikipedia.org/wiki/Woodbury_matrix_identity. Can be derived by solving $(A + UCV)X = I$. Nice!

$$
\begin{align}
X &= (I-\gamma P\pi)^{-1}U(C^{-1}+ V(I-\gamma P\pi)^{-1}U)V(I-\gamma P\pi)^{-1} \\
\epsilon &= X r \pi \\
\epsilon[i] &\le \parallel Xr\pi \parallel_{\infty}
\end{align}
$$

What is the goal here? To write the error in terms of properties of $P, r, \pi$. The condition of $P$, the ...?


TODO. Could visualise this using 2-state 2-action MDP.

![The effect of adding noise to the transition function and then evaluating the policy under the approximately correct estimate.](../pictures/figures/noisy-transitions.png)

## Learning (dynamics / Complexity)

- Partitioning policy space.
- jumping from corner to corner
- two $\epsilon$ different policies. why do they have distinctly different dynamics?!
- number of steps (in high n_actions?)


__Q:__ (how) Does it help?

Complexity of finding the abstraction + solving vs solving ground problem.


Want to show that. The convergence rate / amount of data required is less for the abstraction.

$$
\begin{align}
e_{G} &= \mathcal O(n, |A|, t) \\
e_{A} &= \mathcal O(m, |A|, t) \\
\therefore & e_{G} \ge e_{A}
\end{align}
$$


## Incremental updates

Given $P_0, R_0$.
At each time step we recieve $\Delta R, \Delta P$.
Want to efficiently calculate $\Delta V$ s.t. $V_t = \Delta V + V_{t-1}$.
Oh... Gradient descent.

So how much computation does this incremental update save us?
Are there more efficient ways?

But want to use the information as soon as possible. Offline training means that the data can be 'stale'. Can this be quantified.
Related to distributed optimisation!?

$$
dV = (I-\gamma dP \pi)^{-1}dr \pi
$$



***

- What about exploration?!
- What can go wrong with abstracting? Imagine a simple linear cts relationship between state and value. We then discretise this and must learn the value independently for each of the discretised points. Can be much worse...
