

$$
\zeta = \{(s_t, a_t, r_t): t \in [0, H]\} \\
R(\zeta) = \sum_{t=0}^H \gamma^t \zeta[t, r] \\
D(\zeta, d_i, \pi) = d_i \prod_{t=0}^H \pi(\zeta[t, a] | \zeta[t, s]) \tau(\zeta[t+1, s]| \zeta[t, s], \zeta[t, a]) \\
V^{\pi}(s_0) = \mathop{\mathbb E}_{D(\zeta, 1_{s_0==1}, \pi)} [R(\zeta)] \\
$$
