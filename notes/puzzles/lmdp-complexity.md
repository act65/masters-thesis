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
