Solving an MDP using LMDPs requires three steps.

1. Find a linear embedding of the MDP.
    - $p, q$ s.t. $r(s, a) = q(s) - \text{KL}\Big(P(\cdot | s, a)\parallel p(\cdot | s)\Big)$.
2. Solve the LMDP.
    - $z = QPz^{\alpha}$, $u^{* }(s' | s) = \frac{p(s'|s)z(s')}{\sum_{ \tilde s} p(\tilde s|s)v(\tilde s)}$.
3. Decode the optimal LMDP control.
    - $P_{\pi} = \sum_a P(\cdot | s, a) \pi(a | s), \;\; \pi(a|s) = \mathop{\text{argmin}}_{\pi} \text{KL}\Big(P_{\pi}(\cdot | s)\parallel p(\cdot | s))\Big)$

__Question:__ What is the complexity of each part? How does this compare to a traditional MDP?


## Linear embedding



## Solve the LMDP



## Decoding

Use KL.
Not sure if this makes sense? Should use OT instead?


***

Note. Is there another way to learn $p, q$? In the setting where we are not given access to $\tau, r$, could we learn them instead?
