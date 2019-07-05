In [Policy invariance under reward transformations](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf) the authors prove that __TODO__... explain.

The next question is, what other transforms is the optimal policy invariant to?
- transformations of the transition fn
- transformations of both the reward and transition fn
- anything else?s

$$
\pi^{* } = \mathop{\text{max}}_{\pi}  \mathop{\mathbb E}_{s\sim d_{\pi}}[ V^{\pi}(s) ] \\
V^{\pi}(s) = \mathop{\mathbb E}_{\tau \sim d_{\pi}(s)} [\sum^{\infty}_{t=0} \gamma^t r_t ] \\
d_{\pi}(s) = \sum_{t=0}^{\infty} \gamma^t P(s_t=s| \pi, d_0, T)
$$


Want to find the invariant transformations under the optimal policy.

$$
r'(s, a) = f_r(r(s, a)) \\
\tau'(s, a) = f_{\tau}(\tau(s, a)) \\
r'(s, a), \tau'(s, a) = f_{r,\tau}(r(s, a), \tau(s, a)) \\
\text{s.t. if} \;\; \\
\pi^{* } = \\
\implies \pi^{* } = \\
$$
