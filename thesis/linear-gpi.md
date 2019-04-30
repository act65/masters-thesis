> Want an abstraction that makes the problem easier to solve.

What about an abstraction that reduces the problem to a linear space? That seems nice.
- We know how to solve linear systems of equations with $O()$.
- And we know how to use these solutions to calculate the optimal, policy: generalised policy iteration.

## Linear representations and policy iteration

$$
\begin{align}
V &= r_{\pi} + \gamma P_{\pi} V \\
V - \gamma P_{\pi} V &= r_{\pi}\\
(I-\gamma P_{\pi})V &= r_{\pi}\\
V &= (I-\gamma P_{\pi})^{-1}r_{\pi}\\
\end{align}
$$

Evaluation is normally expensive. But here ...

## Learning the linear abstraction

Describe algol.

The learning reduces to supervised learning. We are provided with examples $(s_t, a_t, r_t, s_{t+1})$

$$
\begin{align}
\textbf  r \in \mathbb R^{n \times m}, &\; \textbf P \in [0,1]^{n \times m \times n} \\
L_{r} &= \text{min} \parallel r_t - \textbf r[\phi(s_t), a_t] \parallel^2_2 \tag{mean squared error}\\
L_{P} &= \mathop{\text{max}}_{\theta} \textbf P[\phi(s_{t+1}),\phi(s_t), a_t]\tag{max likelihood}\\
\end{align}
$$


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
