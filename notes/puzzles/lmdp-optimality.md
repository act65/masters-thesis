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


$$
Q = r + \gamma P\cdot_{(s')} V \\
Q = (I - \gamma P_{\pi})^{-1} r ??? \\
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


***


$$
\begin{align}
\epsilon &=\parallel V_{\pi^{* }} - V_{\pi_{u^{* }}} \parallel_{\infty}  \tag{1}\\
&=\parallel (I - \gamma P_{\pi^{* }})^{-1}r_{\pi^{* }} - (I - \gamma P_{\pi_{u^{* }}})^{-1}r_{\pi_{u^{* } }} \parallel_{\infty} \tag{2}\\
&\le\parallel (I - \gamma P_{\pi^{* }})^{-1}r - (I - \gamma P_{\pi_{u^{* }}})^{-1}r \parallel_{\infty} \tag{3}\\
&=\parallel \bigg((I - \gamma P_{\pi^{* }})^{-1} - (I - \gamma P_{\pi_{u^{* }}})^{-1} \bigg) r \parallel_{\infty} \tag{4}\\
&\le r_{\text{max}} \parallel (I - \gamma P_{\pi^{* }})^{-1} - (I - \gamma P_{\pi_{u^{* }}})^{-1}   \parallel_{\infty} \tag{5}\\
&= r_{\text{max}} \parallel \sum_{t=0}^{\infty} \gamma^t P_{\pi^{* }} - \sum_{t=0}^{\infty} \gamma^t P_{\pi_{u^{* }}}  \parallel_{\infty} \tag{6}\\
&= r_{\text{max}} \parallel \sum_{t=0}^{\infty} \gamma^t (P_{\pi^{* }} - P_{\pi_{u^{* }}})   \parallel_{\infty} \tag{7}\\
&= \frac{r_{\text{max}}}{1-\gamma} \parallel P_{\pi^{* }} - P_{\pi_{u^{* }}} \parallel_{\infty} \tag{7}\\
\end{align}
$$

(1) We want to compare the optimal policies value and the value achieved by the optimal LDMP solution.
(2) Assume that there exists a policy that can generate the optimal control dynamics (as given by the LMDP). In that case we can set $P_{\pi_{u^{* }}} = U^{* }$.
(3) $r_{u^{* }}$ doesnt really make sense as the reward is action dependent. We could calculate it as $r_{\pi_{u^{* } }}$, but we dont explicity know $\pi_{u^{* }}$. $(I - \gamma P_{\pi^{* }})^{-1}r$ represents the action-values, or $Q$ values. By doing this exhange, we might over estimate the diffference under the infinity norm as two non-optimal actions may have larger difference. Also, use the element wise infinity norm.

Notes

- why are we using the infinity norm?!!
-

What does $\delta \ge \text{KL}(P_{\pi^{* }} \parallel U^{* })$ imply about $\parallel P_{\pi^{* }} - P_{\pi_{u^{* }}} \parallel_{\infty}$?
