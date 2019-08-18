Approximate the optimal transition dynamics using the actions we have available.

$$
\begin{align}
P_{\pi}(\cdot | s) = \sum_a P(\cdot | s, a) \pi(a | s) \\
\pi_{u^{* }} = \mathop{\text{argmin}}_{\pi} \sum_s \text{KL}\Big(u^{* }(\cdot | s))\parallel P_{\pi}(\cdot | s)\Big) \\
\text{Or} \\
\pi_{u^{* }} =  \sum_s \mathop{\text{argmin}}_{\pi} \text{KL}\Big(u^{* }(\cdot | s))\parallel P_{\pi}(\cdot | s)\Big) + H(P_{\pi}(\cdot | s))\\
\end{align}
$$

$$
\begin{align}
&= - \sum_s \sum_{s'} u^{* }(s' | s) \log \frac{\sum_a P(s' | s, a) \pi(a | s)}{u^{* }(s' | s)} \\
&= - \sum_s \sum_{s'} u^{* }(s' | s) \log \Big(\sum_a P(s' | s, a) \pi(a | s) \Big)+ u^{* }(s' | s) \log u^{* }(s' | s) \\
\end{align}
$$

***

$$
r(s, a) = q(s) - \sum_{s'} P(s'|s, a) \log p(s' | s)
$$

- If $P(s'|s, a)$ is zero, then $p(s'|s)$ can be whatever it likes...
- If $p(s'|s)$ is $1$ then it doesnt matter what $P(s'|s, a)$ as $\log 1 = 0$.
- Rewards for impossible actions. Doesnt make sense. Since transition p is 0, the reward must be q!?

***

For $|A|<|S|$, the transitions from a given state are low rank. We are restricted to a subspace of distributions over $S$ we can pick. This distribution is the connectivity.
A subspace where we are allowed to put mass, everywhere else is zero.

Assuming we can perfectly minimise the KL in each subspace. What does that imply about the total KL?


Need to exploit $p$. Information about the transition dynamics have constrained the optimal policy!!!


Strategies
- Consider in cts domain, and exploit locality
- Mess around with Bayes to get conditionals playing nicely

If $P(s'| s, a) = 0$.
Assume $$
