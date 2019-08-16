#### Optimality of solutions via LMDPs

> Do these two paths lead to the same place?
<!-- insert quote?! -->

One of the main questions we have not addressed yet is; if we solve the MDP directly, or solve it via our linear abstraction (linearise, solve and project), do we end up in the same place? This is a question about the completeness of our abstraction. Can our abstraction represent (and find) the same solutions that the original can?

Why does this matter? When we apply our linear abstraction, we want to know: can I trust the answer it has given? If I follow the actions specified by optimal policy, am I going to get rewards?

<!-- As a warm up. Let's compare value returned by optimal vs random -->

$$
\begin{align}
\parallel V_{\pi^{* }} - V_{\pi_{u^{* }}} \parallel_{\infty}&= \epsilon  \tag{1}\\
&=\parallel (I - \gamma P_{\pi^{* }})^{-1}r_{\pi^{* }} - (I - \gamma P_{\pi_{u^{* }}})^{-1}r_{\pi_{u^{* } }} \parallel_{\infty} \tag{2}\\
&\le\parallel (I - \gamma P_{\pi^{* }})^{-1}r_{\text{max}} - (I - \gamma P_{\pi_{u^{* }}})^{-1}r_{\text{min}} \parallel_{\infty} \tag{3}\\
&=\parallel \bigg((I - \gamma P_{\pi^{* }})^{-1} - (I - \gamma P_{\pi_{u^{* }}})^{-1} \bigg) \Delta r \parallel_{\infty} \tag{4}\\
&\le \Delta r_{\text{max}} \parallel (I - \gamma P_{\pi^{* }})^{-1} - (I - \gamma P_{\pi_{u^{* }}})^{-1}   \parallel_{\infty} \tag{5}\\
&= \Delta r_{\text{max}} \parallel \sum_{t=0}^{\infty} \gamma^t P_{\pi^{* }} - \sum_{t=0}^{\infty} \gamma^t P_{\pi_{u^{* }}}  \parallel_{\infty} \tag{6}\\
&= \Delta r_{\text{max}} \parallel \sum_{t=0}^{\infty} \gamma^t (P_{\pi^{* }} - P_{\pi_{u^{* }}})   \parallel_{\infty} \tag{7}\\
&= \frac{\Delta r_{\text{max}}}{1-\gamma} \parallel P_{\pi^{* }} - P_{\pi_{u^{* }}} \parallel_{\infty} \tag{8}\\
\end{align}
$$

(1) We want to compare the value achieved by the optimal policy and the value achieved by the optimal linearised solution.
(2) Assume that there exists a policy that can generate the optimal control dynamics (as given by the LMDP). In that case we can set $P_{\pi_{u^{* }}} = U^{* }$.
(3) $r_{u^{* }}$ doesnt really make sense as the reward is action dependent. We could calculate it as $r_{\pi_{u^{* } }}$, but we dont explicity know $\pi_{u^{* }}$. $(I - \gamma P_{\pi^{* }})^{-1}r$ represents the action-values, or $Q$ values. By doing this exhange, we might over estimate the diffference under the infinity norm as two non-optimal actions may have larger difference. Also, use the element wise infinity norm.

(3) Let's assume the optimal policy picks $\mathop{\text{max}}_a r(s, a)$ at every step. Then the worst case is that we pick $\mathop{\text{min}}_a r(s, a)$. Write $\mathop{\text{max}}_a r(s, a)- \mathop{\text{min}}_a r(s, a) = \Delta r$
(4) No. Cannot do that... Assumes $(I - \gamma P_{\pi^{* }})^{-1} = (I - \gamma P_{\pi_{u^{* }}})$. Or that the eqn can be factored.



Notes

- why are we using the infinity norm?!!
- What does $\delta \ge \text{KL}(U^{* } \parallel P_{\pi_{u^{* } }} )$ imply about $\parallel P_{\pi^{* }} - P_{\pi_{u^{* }}} \parallel_{\infty}$?
- Is it possible to relate $U^{* }$ to $\pi^{* }$? $\text{KL}(U^{* } \parallel P_{\pi^{* }} )$
- Use $r \in [0, 1]$. Like in other papers. This would simplify and allow us to do (4)?? But need to check it generalises.

***

Alternative.

$$
\begin{align}
\parallel V_{\pi^{* }} - V_{\pi_{u^{* }}} \parallel_{\infty}&= \epsilon  \tag{1}\\
=\parallel \mathop{\text{max}}_a \bigg[ r(s, a) + \gamma \mathop{\mathbb E}_{s'\sim P(\cdot|s, a)} [V(s')] \bigg] &- \mathop{\mathbb E}_{a\sim \pi_{u^{* }}(\cdot | s)} r(s, a) - \gamma \mathop{\mathbb E}_{s'\sim \sum_aP(\cdot|s, a)\pi_{u^{* }}(a | s)}  [V(s')]  \parallel_{\infty} \tag{2}\\
=\parallel \mathop{\text{max}}_a \bigg[ r(s, a) - \mathop{\mathbb E}_{\tilde a\sim \pi_{u^{* }}(\cdot | s)} r(s, \tilde a) &+ \gamma \mathop{\mathbb E}_{s'\sim P(\cdot|s, a)} [V(s')]  - \gamma \mathop{\mathbb E}_{s'\sim \sum_{\tilde a}P(\cdot|s, \tilde a)\pi_{u^{* }}(\tilde a | s)}  [V(s')] \bigg] \parallel_{\infty} \tag{2}\\
\end{align}
$$


***

Relies a lot on the result of the projection from optimal state distributions to state-action policies.

\begin{align}
P_{\pi}(\cdot | s) = \sum_a P(\cdot | s, a) \pi(a | s) \\
\pi = \mathop{\text{argmin}}_{\pi} \text{KL}\Big(u(\cdot | s))\parallel P_{\pi}(\cdot | s)\Big) \\
\delta = \text{KL}\Big(u^{* }(\cdot | s))\parallel P_{\pi}(\cdot | s)\Big) \\
\delta = - \sum_{s'} u^{* }(s'|s) \log \frac{P_{\pi}(s'|s)}{u^{* }(s' | s)}
\end{align}

Small delta implies ???.
Everywhere $u^{* }$ has non-zero probability, $P_{\pi}(s'|s)$, is $\approx u(s'|s)$


KL

>  the expected number of extra bits required to code samples from $P$ using a code optimized for $Q$ rather than the code optimized for $P$

Always gives more weight to the high probability (under u) states. And because the two distributions are normalised. For all mass that $P_{\pi}>u$, there will be equal mass where $P_{\pi}<u$. But the latter is weighted more. (kinda)


***

want to know the difference between the two policies.

\begin{align}
\delta = \text{KL}\Big(\pi^{* }\parallel \pi_{u^{* }} \Big) \\
\end{align}

Could then use this to relate $r_{\pi^{* }}$ and $r_{\pi_{u^{* }}}$? But how!?!
