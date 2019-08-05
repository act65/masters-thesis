What objective are we optimising?
How can we characterise this optimisation problem?

What does it mean to behave optimally?
How do we measure it?

$$
\begin{align*}
V(\pi^{* }) \equiv  \mathop{\mathbb E}_{s_0\sim d_0} \mathop{\text{max}}_{a_0} r(s_0, a_0)
+ \gamma  \mathop{\mathbb E}_{s_1\sim p(\cdot | s_0, a_0)} \Bigg[ \\ \mathop{\text{max}}_{a_1} r(s_1, a_1)
+ \gamma \mathop{\mathbb E}_{s_2\sim p(\cdot | s_1, a_1)} \bigg[ \\ \mathop{\text{max}}_{a_2} r(s_2, a_2)
+ \gamma  \mathop{\mathbb E}_{s_3\sim p(\cdot | s_2, a_2)} \Big[
\dots \Big] \bigg] \Bigg]
\end{align*}
$$


$$
\zeta = \{(s_0, a_0), (s_1, a_1), \dots (s_N, a_N)\} \\
P(\zeta, \pi, \tau, d_0) = d_0(s_0)\Pi_{t=0}^T \tau(s_{t+1} | s_t, a_t)\pi(a_t|s_t) \\
V(\zeta, r, \gamma) = \sum^T_{t=0} \gamma^t r(s_t, a_t) \\
\pi^{* } = \mathop{\text{argmax}}_{\pi} \;\;  \mathop{\mathbb E}_{\zeta \sim P(\cdot, \pi, \tau, d_0)} V(\zeta, r, \gamma)\\
$$
