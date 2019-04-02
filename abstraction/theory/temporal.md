## Transition based abstraction for transfer

Abstraction for model based RL!? Learning abstract models of the transition function.

As noted above. Using the $Q$ function or the reward function to construct an abstraction makes the abstraction task specific.

$$
\forall_{s_A \in S_A} \mid \sum_{s_G' \in G(s_A)} \tau(s_1, a, s_G') - \tau_G(T(s_2), T(a),T(s_G'))\mid\le \epsilon
$$

## Temporal abstraction

$$
\begin{align}
Q(s_{1:m}, a) &= \sum_1^{m-1} \gamma^t r(s_t) + \gamma^mQ(s_m, a)\\
\phi (s_{1:m}) &= \phi(x_{1:n}) \implies \mid Q(s_{1:m}, a) - Q(x_{1:n}, a)\mid \le \epsilon \tag{Temporal state abstraction}\\
\\
Q(s, a_{1:m}) &= \sum_1^{m-1} \gamma^t r(s_t, a_t) + \gamma^mQ(s_m, a_m)\\
\phi (a_{1:m}) &= \phi(b_{1:n}) \implies \mid Q(s, a_{1:m}) - Q(s, b_{1:n})\mid \le \epsilon \tag{Temporal action abstraction}\\
\end{align}
$$

Can be framed as an unusual type of action abstraction, where the meaning of abstraction is being stretched...
Initially we had $Q(s,a)\quad s\in S, a\in A$ but we have transformed this (in the case of temporal-action abstraction) to $Q(s,a)\quad s\in S, a\in A^{* }$.
