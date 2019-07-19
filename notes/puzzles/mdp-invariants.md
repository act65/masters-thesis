In [Policy invariance under reward transformations](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf) the authors prove that;

> The set of transforms, $T$, of the reward function, that yield the same optimal policy, $\pi^{* }$ are of the form; $T(R)(s, a, s') = R(s, a, s')+F(s, a, s')$ where $F$ is a potential function, $F(s, a, s') = \gamma \phi(s') - \phi(s)$.

The next question is, what about other transforms of the MDP that yield the same optimal policy?
- transformations of the transition fn
- transformations of both the reward and transition fn
- transformations of the reward and discount


Want to find the invariant transformations under the optimal policy.

$$
r'(s, a) = f_r(r(s, a)) \\
\tau'(s, a) = f_{\tau}(\tau(s, a)) \\
r'(s, a), \tau'(s, a) = f_{r,\tau}(r(s, a), \tau(s, a)) \\
\text{s.t. if} \;\; \\
\pi^{* } = \\
\implies \pi^{* } = \\
$$


$$
\zeta = \{(s_0, a_0), (s_1, a_1), \dots (s_N, a_N)\} \\
P(\zeta, \pi, \tau, d_0) = d_0(s_0)\Pi_{t=0}^T \tau(s_{t+1} | s_t, a_t)\pi(a_t|s_t) \\
V(\zeta, r, \gamma) = \sum^T_{t=0} \gamma^t r(s_t, a_t) \\
\pi^{* } = \mathop{\text{argmax}}_{\pi} \;\;  \mathop{\mathbb E}_{\zeta \sim P(\cdot, \pi, \tau, d_0)} V(\zeta, r, \gamma)\\
$$

^^^ This formulation makes it clearer that the policy can only move probabilities around. It cannot change the rewards. It can just weight them differently. Cannot change the space of possible trajectories, we can only make some more likely than others.

Want to find $T$ such that;

$$
\begin{align}
\pi^{* } &= \mathop{\text{argmax}}_{\pi} \;\;  \mathop{\mathbb E}_{\zeta \sim P(\cdot, \pi, \tau, d_0)} V(\zeta, r, \gamma)\\
&= \mathop{\text{argmax}}_{\pi} \;\;  \mathop{\mathbb E}_{\zeta \sim P(\cdot, \pi, T(\tau), d_0)} V(\zeta, r, \gamma)\\
\end{align}
$$


$$
\begin{align}
\pi^{* } &= \mathop{\text{argmax}}_{\pi} \;\;  \mathop{\mathbb E}_{\zeta \sim P(\cdot, \pi, \tau, d_0)} V(\zeta, r, \gamma)\\
&= \mathop{\text{argmax}}_{\pi} \;\;  \mathop{\mathbb E}_{\zeta \sim P(\cdot, \pi,\tau, d_0)} V(\zeta, r+f, \gamma) \\\text{ iff } f(s, a) &= \gamma \mathop{\mathbb E}_{s' \sim \tau(s,a)}[\Phi(s')] - \Phi(s)\\
\end{align}
$$


Intuition.

Properties of $\tau$ that are sufficient...?
- Dont delete existing transitions: If $p(s' | s, a) > 0$, then $T(p)(s' | s, a) > 0$
- Dont add new transitions: If $p(s' | s, a) = 0$, then $T(p)(s' | s, a) = 0$
- If , the optimal policy will pick $a_1$. Therefore $\pi^{* }$ the same for all transitions that satisfy $\{ \tau : V(s') \cdot \tau(s' | s, a_1) > V(s') \cdot \tau(s' | s, a_2)\}$

If $Q(s, a_1)> Q(s,a_2)$ then the optimal policy will pick $a_1$.

$$
r(s, a_1) + \gamma \mathop{\mathbb E}_{s\sim\tau(\cdot|s, a_1)}V(s) > r(s, a_2) + \gamma \mathop{\mathbb E}_{s\sim\tau(\cdot|s, a_1)} V(s) \\
\mathcal T = \{\tau : \}
$$


$$
Q = r + \gamma \text{max}_{\pi} P_{\pi}V \\
\delta_{ijk} = Q(s_i, a_j) - Q(s_i, a_k)
$$

Any transformation that preserves the argmax of $\delta_i$

can shuffle probability in cycles between the other sub optimal actions.

***
