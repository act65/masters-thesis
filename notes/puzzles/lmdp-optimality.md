
Define
$$
\pi_u(a|s) = \mathop{\text{argmin}}_{\pi} \text{KL}\Big(P_{\pi}(\cdot | s)\parallel u(\cdot | s))\Big)\\
V^{\pi}_M(s) = \mathop{\mathbb E}_{a \sim \pi(a|s)}[r(s, a)] + \gamma \mathop{\mathbb E}_{s' \sim \int \pi(a|s)\tau(s'|s,a)da}[V(s')] \\
$$

__Question:__ How suboptimal are LMDP solutions to MDPs?

$$
\parallel V^{\pi^* }_M - V^{\pi_{u^* }}_M \parallel = \epsilon
$$
