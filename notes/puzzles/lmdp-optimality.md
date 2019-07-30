__Question:__ How suboptimal are LMDP solutions to MDPs?

$$
\parallel V^{\pi^* }_M - V^{\pi_{u^* }}_M \parallel \le \epsilon
$$

Where, we have the definitions;

$$
\begin{align}
\pi_u(a|s) = \mathop{\text{argmin}}_{\pi} \text{KL}\Big(P_{\pi}(\cdot | s)\parallel u(\cdot | s))\Big)\\
u^{* }(s' | s) = \frac{p(s'|s) z(s')}{\sum_{s'} p(s'|s) z(s')}\\
V^{\pi}_M(s) = \mathop{\mathbb E}_{a \sim \pi(a|s)}[r(s, a)] + \gamma \mathop{\mathbb E}_{s' \sim \int \pi(a|s)\tau(s'|s,a)da}[V(s')] \\
\end{align}
$$

???
***

$$
\pi_u(a|s) = \mathop{\text{argmin}}_{\pi} \text{KL}\Big(\sum_aP(\cdot | s, a)\pi(a|s)\parallel  \frac{p(\cdot|s) z(\cdot)}{\sum_{s'} p(s'|s) z(s')})\Big)\\
$$

1) Need to write $u^{* }(\cdot|s) = \frac{p(\cdot|s) z(\cdot)}{\sum_{s'} p(s'|s) z(s')}$  in terms of $v, r, P$!?
    - But we might not have a perfect embedding.
2) Derive the form of $\pi_u$
3) Derive the value of the optimal LMDP policy, $V_M^{\pi_{u^{* }}}$.
    - But knowing the a policy doesnt help us bound the value? As that depends on the evironment?!?

***

$$
\parallel Q^{\pi^* }_M - Q^{\pi_{u^* }}_M \parallel_{\infty} \le \epsilon \\
\parallel \bigg[ r(s, a) +\gamma \mathop{\text{max}}_{a'} \mathop{\mathbb{E}}_{s'\sim \tau(\cdot | s, a)} Q(s',a')\bigg] - V^{\pi_{u^* }}_M \parallel_{\infty}
$$

$$
\parallel V^{\pi^* }_M - V^{\pi_{u^* }}_M \parallel_{\infty} \le \epsilon \\
\parallel \mathop{\text{max}}_{a} \bigg[ r(s, a) +\gamma \mathop{\mathbb{E}}_{s'\sim \tau(\cdot | s, a)} V(s')\bigg] - V^{\pi_{u^* }}_M \parallel_{\infty}
$$

***

Need to relate the value of the LMDP to the value of the MDP.
If $V_{LMDP} = -\log(z)$ then $V_{MDP} \le / \approx f(V_{LMDP})$ ...?
