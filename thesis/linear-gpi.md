> Want an abstraction that makes the problem easier to solve.

What about an abstraction that reduces the problem to a linear space? That seems nice.
- We know how to solve linear systems of equations with $O()$.
- And we know how to use these solutions to calculate the optimal, policy: generalised policy iteration.

## Linear evaluation and policy iteration

Evaluation is normally expensive. But here ...

$$
\begin{align}
V &= r_{\pi} + \gamma P_{\pi} V \tag{bellman eqn}\\
V - \gamma P_{\pi} V &= r_{\pi}\\
(I-\gamma P_{\pi})V &= r_{\pi}\\
V &= (I-\gamma P_{\pi})^{-1}r_{\pi}\\
\end{align}
$$

(finding the optimal policy is still a non-linear problem. how is it non-linear?!)

Alternative derivation. From neumann series and bellman operator.

$$
\begin{align}
(I -T)^{-1} &= \sum^{\infty}_{t=0} T^k \\
T &= r_{\pi} + \gamma P_{\pi} \\
???
\end{align}
$$

If the Neumann series converges in the operator norm (or in any norm?), then I – T is invertible and its inverse is the series.
If operator norm of T is < 1. Then will converge to $(I-T)^{-1}$?


## Learning the (linear) abstraction

Describe algol.

The learning reduces to supervised learning. We are provided with examples $(s_t, a_t, r_t, s_{t+1})$

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
\hat P &= P + E \\
\hat V &= (I-\gamma \hat P \pi)^{-1}r \pi \\
\epsilon &= V - \hat V\\
&= (I-\gamma P \pi)^{-1}r \pi -  (I-\gamma (P + E) \pi)^{-1}r \pi \\
&= \Big((I-\gamma P \pi)^{-1} -  (I-\gamma P\pi + \gamma E\pi)^{-1} \Big)r \pi \\
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

## Theory
### Understanding learning (dynamics / Complexity)

- Partitioning policy space.
- jumping from corner to corner
- two $\epsilon$ different policies. why do they have distinctly different dynamics?!
- number of steps (in high n_actions?)


__Q:__ (how) Does it help?

Complexity of finding the abstraction + solving vs solving ground problem.

### Approximate abstractions

__Q:__ What about evaluating a policy with approximately accurate $P, r$?

$$
V'(\pi) = (I-\gamma (P+\delta_p)\pi)^{-1}(r+\delta_r)\pi \\
V(\pi^{* }) - V'(\pi) \le \epsilon \\
$$
