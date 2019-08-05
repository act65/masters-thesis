__Question:__ How suboptimal are LMDP solutions to MDPs?

$$
\begin{align}
&\parallel V_{\pi^{* }} - V_{\pi_{u^{* }}} \parallel_{\infty} \le \epsilon \\
\end{align}
$$

Potential solution sketches

- use q values!?
- solve $\pi_{u^{* }}$ in terms of $P, r$. Estimate its value relative to $\pi^{* }$.
- ??!?

***

$$
\parallel Q^{\pi^* }_M - Q^{\pi_{u^* }}_M \parallel_{\infty} \le \epsilon \\
\parallel \bigg[ r(s, a) +\gamma \mathop{\text{max}}_{a'} \mathop{\mathbb{E}}_{s'\sim \tau(\cdot | s, a)} Q(s',a')\bigg] - V^{\pi_{u^* }}_M \parallel_{\infty}
$$

$$
\parallel V^{\pi^* }_M - V^{\pi_{u^* }}_M \parallel_{\infty} \le \epsilon \\
\parallel \mathop{\text{max}}_{a} \bigg[ r(s, a) +\gamma \mathop{\mathbb{E}}_{s'\sim \tau(\cdot | s, a)} V(s')\bigg] - V^{\pi_{u^* }}_M \parallel_{\infty}
$$



#### Near optimal LMDPs


Want to solve the above for $\epsilon$.
$$
\begin{align}
\pi_{u^{* }} &= \mathop{\text{argmin}}_{\pi} \sum_s\text{KL}(u^{* }(\cdot|s)\parallel \mathop{\mathbb E}_{a\sim \pi_{u^{* }}(\cdot | s) }P(\cdot | s, a)) \\
\mathop{\mathbb E}_{a\sim \pi_{u^{* }}(\cdot | s) }P(\cdot | s, a) &= u^{* }(\cdot|s) \;\; \forall s\in S\\
\mathop{\mathbb E}_{a\sim \pi_{u^{* }}(\cdot | s) }P(\cdot | s, a) &= \frac{p(\cdot | s)\cdot z^{* }(\cdot)^{\gamma}}{\sum_{s'} p(s' | s) z^{* }(s')^{\gamma}} \;\; \forall s\in S\\
P\pi_{u^{* }} &= G^{-1}p{z^{* }}^{\gamma} \\
\pi_{u^{* }} &= P^{-1}G^{-1}p{z^{* }}^{\gamma} \\
\end{align}
$$
