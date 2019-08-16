Solving an MDP using LMDPs requires three steps.

1. Find a linear embedding of the MDP.
    - $p, q$ s.t. $r(s, a) = q(s) - \text{KL}\Big(P(\cdot | s, a)\parallel p(\cdot | s)\Big)$.
2. Solve the LMDP.
    - $z = QPz^{\alpha}$, $u^{* }(s' | s) = \frac{p(s'|s)z(s')}{\sum_{ \tilde s} p(\tilde s|s)v(\tilde s)}$.
3. Decode the optimal LMDP control.
    - $\pi(a|s) = \mathop{\text{argmin}}_{\pi} \text{KL}\Big(P_{\pi}(\cdot | s)\parallel p(\cdot | s))\Big), \;\; P_{\pi} = \sum_a P(\cdot | s, a) \pi(a | s)$

__Question:__ What is the complexity of each part? How does this compare to a traditional MDP?


## Linear embedding

```python
for state in states:

```

$\mathcal O (|S| \times (|S||A|/2)^{2.3}  )$


## Solve the LMDP

$\mathcal O ((|S|^{2.3}  )$

## Decoding

$\mathcal O( (|S||A|/2)^{2.3} )$


$$
\begin{align}
P_{\pi}(\cdot | s) &= \sum_a P_k(\cdot | s, a) \pi(a | s) \\
\pi &= \mathop{\text{argmin}}_{\pi} \text{KL}\Big(u(\cdot | s))\parallel P_{\pi}(\cdot | s)\Big) \\
0 &=\text{KL}\Big(u(\cdot | s))\parallel P_{\pi}(\cdot | s)\Big) \\
 &= -\sum_{s'} u(s' | s) \log\frac{P_{\pi}(s' | s)}{u(s' | s)}  \\
 \begin{cases}
    x(n), & \text{for } 0 \leq n \leq 1 \\
    x(n - 1), & \text{for } 0 \leq n \leq 1 \\
    x(n - 1), & \text{for } 0 \leq n \leq 1
  \end{cases}
\end{align}
$$
