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
