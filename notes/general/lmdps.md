Questions

- What is p(s'|s)!?!?
- Want some examples of MDPs they cannot solve.
- What is the relationship to other action embedding strategies?
- How does p(s'|s) bias the controls found??? I can imagine the unconstrained dynamics acting as a prior and prefering some controls over others.
- If we have m states and n actions. Where m >> n. Then $u(s'|s)$ is much larger than $\pi(a|s)$. Also, $u(s'|s)$ should be low rank?! $u_{s's} = \sum_a u_a \alpha_a u_a^T$

## Assumptions

Two main transformations of the LMDP, everything else follows.

1. Allow the direct optimisation of transitions, $u(s'|s)$, rather than policies.
1. $r(s, a) = q(s) + KL(P(\cdot|s, a)\parallel p(s'|s)), \forall s, a$

Another way to frame. __Q:__ If we want to optimise the space of transitions, what augmentations of the MDP are necessary to ensure solutions in the LMDP are optimal in the MDP?

- Prove that 2. is necessary and sufficient for optimality. (probs not possible?!)


Pick $a \in A$, versus, pick $\Delta(S)$. $f: S\to A$ vs $f:S \to \Delta(S)$.


## Option decoding

What about using options to help solve the optimal control decoding?

$$
P_{\pi} = \sum_\omega P_k(\cdot | s, \omega) \pi(\omega | s) \\
\pi = \mathop{\text{argmin}}_{\pi} \text{KL}\Big(u(\cdot | s))\parallel P_{\pi}(\cdot | s)\Big)
$$

Options would allow greater flexibility in the $P_{\pi}(\cdot | s)$ distribution, making is possible to match $u(s'|s)$ with greater accuracy (and possibly cost).

- First need to demonstrate that action decoding is lossy.
- Then show that using options is less lossy.

## Embedding

problem with embedding..

$$
q(s) - KL(P(\cdot | s, a) \parallel p(\cdot | s)) = r(s, a) \\
$$

The KL. If $P(s' | s, a)$ is zero, then $p(s' | s)$ can be whatever it likes. Thus, $p(x' | x)$ might contain many impossible transitions.


## Heirarchical


...


## Unconstrained dynamics

- What is their function?
- What do they look like?



## Maximisation derivation

$$
\begin{align}
r(s, a) &= q(s) - \text{KL}(u(\cdot | s) \parallel p(\cdot | s)) \\
V(s) &= \mathop{\text{max}}_{a} r(s, a) + \gamma \mathop{\mathbb E}_{s' \sim p(\cdot | s, a)} V(s') \tag{mdp}\\
\\
\hat V(s) &= q(s) + \mathop{\text{max}}_{u}   \gamma \mathop{\mathbb E}_{s' \sim u(\cdot | s)} \big[V(s')\big]  - \text{KL}(u(\cdot | s) \parallel p(\cdot | s)) \tag{lmdp} \\
&= q(s) + \mathop{\text{max}}_{u} \gamma \mathop{\mathbb E}_{s' \sim u(\cdot | s)} \big[V(s')\big]  - \mathop{\mathbb E}_{s' \sim u(\cdot | s)} \log(\frac{p(s' | s) }{ u(s' | s)}\\
z(s) &= e^{-v(s)} \\
&= q(s) + \mathop{\text{max}}_{u} \gamma \mathop{\mathbb E}_{s' \sim u(\cdot | s)} \big[\log(\frac{u(s' | s)}{z(s')p(s' | s)}) \big]\\
\mathop{\mathbb E}_{s' \sim u(\cdot | s)} \big[\log(\frac{u(s' | s)}{z(s')p(s' | s)}) \big]&=\\
u^{* } &= ... \\

\end{align}
$$

...

***
