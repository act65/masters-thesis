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


## Why do we care about LMDPs???

- Transfer
- It efficiently solves the optimal control problem (while introducing other problems...)
- it might provide a way to analyse subgoal techniques!?

LMDPs have the property that if we have already solved two LMDPs, with the same state space, action space, unconditioned transition dynamics, but different state rewards, $q_1, q_2$. Then we can solve a new LMDP, again with the same, ..., and state rewards in the span of $q_1, q_2$, $z_3 = w_1 z_1 + w_2 z_2$, ...

Problem. What does it even mean for two LMDPs to have the same unconditioned dynamics but different state rewards?
The MDPs must have been the same up to some additive constant (constant in the actions), $r(s, a)=r(s, a) + c(s)$.
Does this really capture what we mean by different tasks?!?


## Embedding derivation

Goal. Take a MDP and find a LMDP that has similar structure.

__NOTE:__ The derivation is not the same as in Todorov. He sets $b_a \neq r, b_a = r - \sum P \log P$.

$$
\begin{align}
\forall s, s' \in S, \forall a \in A, \exists u_a& \;\;\text{such that;} \tag{1}\\
P(s' | s, a) &= u_a(s'|s)p(s'|s) \tag{2}\\
r(s, a) &= q(s) - \text{KL}(P(\cdot | s, a) \parallel u_a(\cdot| s) ) \tag{3}\\
\\
r(s, a) &= q(s) - \text{KL}(P(\cdot | s, a)\parallel\frac{P(\cdot | s, a)}{p(\cdot|s)}) \tag{4}\\
r(s, a) &= q(s) - \sum_{s'}P(s' | s, a) \log(p(s'|s)) \tag{5}\\
\\
m_{s'}[s]&:= \log p(s' | s) \tag{6}\\
D_{as'}[s] &:= p(s'|s, a) \tag{7}\\
c_{s'}[s] &:= q[s] \mathbf 1 - m_{s'}[s] \;\;\text{such that} \;\; \sum_{s'} e^{m_{s'}[s]} = 1 \tag{8}\\
\\
r_a &= D_{as'} ( q \mathbf 1 - m_{s'}) \;\;\forall s \tag{9}\\
r_a &= D_{as'}c_{s'}  \;\;\forall s \tag{10}\\
c_{s'} &= r_aD_{as'}^{\dagger} \;\;\forall s\tag{11}\\
q &= -\log \sum_{s'} e^{-c_{s'}} \;\;\forall s\tag{12}\\
m_{s'} &= q - c_{s'} \;\;\forall s\tag{14}\\
\end{align}
$$



Therefore we have $|A|$ linear relationships, for each $s\in S$.

(5) KL introduces a negative, but we also use the log rule to move $p(s'|s)$ to the nominator, adding another negative, and thus cancelling. Yielding the cross entropy between .. and ... .

(11) The More-Penrose pseudo inverse. We apply to the RHS... ???.

> If $D$ is row-rank defficient, the solution $c$ is not unique, and we should be able to exploit the freedom in choosing $c$ to improve the approximation of the traditional MDP. If $D$ is coumn-rank-defficient then an exact embedding cannot be constructed. However, this is unlikely in to occur in practice because it essentially means that the number of symbolic actions is greater than the number of possible next states.

## Unconstrained dynamics

- What is their function?
- What do they look like?

Does it make sense to treat the q(s) like rewards?!
They reward for bing in state s.
But cant capture action specific rewards!?

## Scaling to more complex problems

- sample based / incremental
- large / cts state spaces
- sparse rewards
- ?

### Incremental implementation

Generalise to a more complex problem. We are only given samples.
A first step to tackling more complex problems.

##### Model based
Learn $p, q$ based on samples.

$$
\mathcal L(\theta, \phi) = \mathop{\mathbb E}_{s, a,} \bigg[ r(s, a) - q_\theta(s) + \text{KL}(p_\phi(\cdot | s) \parallel P(\cdot | s, a)) \bigg]\\
\mathcal L(\theta, \phi) = \mathop{\mathbb E}_{s, r, s'} \bigg[r - q_\theta(s) - p_\phi(s' | s) \log \frac{1}{ p_\phi(s' | s)} \bigg] \\
$$

##### Model free

$z$-iterations. But we need to find a way to project $(s_t, a_t, r_t) \to (x_t, p_t, q_t)$.

- Is there a way to construct $p, q$ incrementally!?!?
- What is the essence of what is being done here?

## Maximisation derivation

In the original Todorov paper, they derive the LMDP equations for minimising a cost function. This maximisation derivation just changes a few negative signs around. Although there is also a change in the interpretation of what the unconstrained dynamics are doing. ...?

$$
\begin{align}
V(s) &= \mathop{\text{max}}_{u} q(s) - \text{KL}(u(\cdot| s) \parallel p(\cdot | s)) + \gamma \mathop{\mathbb E}_{s' \sim u(\cdot | s)} V(s') \tag{1}\\
\\
&= q(s) + \mathop{\text{max}}_{u} \bigg[ \mathop{\mathbb E}_{s' \sim u(\cdot | s)} \log(\frac{p(s' | s) }{ u(s' | s)}+\gamma \mathop{\mathbb E}_{s' \sim u(\cdot | s)} \big[V(s')\big] \bigg] \tag{2}\\
\log(z(s)) &= q(s) + \mathop{\text{max}}_{u} \bigg[ \mathop{\mathbb E}_{s' \sim u(\cdot | s)} \log(\frac{p(s' | s) }{ u(s' | s)}+\mathop{\mathbb E}_{s' \sim u(\cdot | s)} \big[\log(z(s')^{\gamma})\big] \bigg] \tag{3}\\
&= q(s) + \mathop{\text{max}}_{u} \bigg[ \mathop{\mathbb E}_{s' \sim u(\cdot | s)} \log(\frac{p(s' | s)z(s')^{\gamma} }{ u(s' | s)} ) \bigg] \tag{4}\\
G(s) &= \sum_{s'} p(s' | s) z(s')^{\gamma} \tag{5}\\
&= q(s) + \mathop{\text{max}}_{u} \bigg[ \mathop{\mathbb E}_{s' \sim u(\cdot | s)} \log(\frac{p(s' | s)z(s')^{\gamma} }{ u(s' | s)} \cdot \frac{G(s)}{G(s)} ) \bigg] \tag{6}\\
&= q(s) + \log G(s) + \mathop{\text{min}}_{u} \bigg[\text{KL}\big(u(\cdot | s) \parallel \frac{p(\cdot | s)\cdot z(\cdot)^{\gamma}}{G(s)} \big) \bigg] \tag{7}\\
u^{* }(\cdot | s) &= \frac{p(\cdot | s)\cdot z(\cdot)^{\gamma}}{\sum_{s'} p(s' | s) z(s')^{\gamma}} \tag{8}\\
\log(z_{u^{* }}(s)) &= q(s) + \log \big(\sum_{s'} p(s' | s) z_{u^{* }}(s')^{\gamma}\big) \tag{9}\\
z_{u^{* }}(s) &= e^{q(s)}\big(\sum_{s'} p(s' | s) z_{u^{* }}(s')^{\gamma}\big) \tag{10}\\
z_{u^{* }} &= e^{q(s)}\cdot P z_{u^{* }}^{\gamma} \tag{11}\\
\end{align}
$$

By definition, an LMDP is the optimisation problem in (1). We can move the $\text{max}$ in (2), as $q(s)$ is not a function of $u$. Also in (2), expand the second term using the definiton of KL divergence, the negative from the KL cancels the second terms negative. (3) Define a new variable, $z(s) = e^{v(s)}$. Also, use the log rule to move the discount rate. (4) Both expectations are under  the same distribution, therefore they can be combined. Also, using log rules, combine the log terms. (5) Define a new variable that will be used to normalise $p(s' | s)z(s')^{\gamma}$. (6) Multiply and divide by $G(s)$. This allows us to rewrite the log term as a KL divergence as now we have two distributions, $u(\cdot | s)$ and $\frac{p(\cdot | s)z(\cdot)^{\gamma}}{G(s)}$. (7) The change to a KL term introduces a negative, instead of maximising the negative KL, we minimise the KL. Also in (7) the extra G(s) term can be moved outside of the expectation as it is not dependent in $s'$. (8) Set the optimal policy to minimise the KL distance term. (9) Since we picked the optimal control to be the form in (8), the KL divergence term is zero. (10) Move the log. (11) Rewrite the equations for the tabular setting, giving a $z$ vector, uncontrolled dynamics matrix.


***

Alternative perspective. The high value trajectories are the most likely ones.

### Distributions over states

What if we wanted to approximate these distributions?
Generalise subgoal methods to work with distributions?
The distribution could be constructed via; parzen window / GMM, neural flow, ?!.

Connections to distributional RL?


## Option decoding

What about using options to help solve the optimal control decoding?

$$
P_{\pi} = \sum_\omega P_k(\cdot | s, \omega) \pi(\omega | s) \\
\pi = \mathop{\text{argmin}}_{\pi} \text{KL}\Big(u(\cdot | s))\parallel P_{\pi}(\cdot | s)\Big)
$$

Options would allow greater flexibility in the $P_{\pi}(\cdot | s)$ distribution, making is possible to match $u(s'|s)$ with greater accuracy (and possibly cost).

- First need to demonstrate that action decoding is lossy.
- Then show that using options is less lossy.


***


Refs

- [Efficient computation of optimal actions](https://www.pnas.org/content/106/28/11478)
- [Linearly-solvable Markov decision problems](https://homes.cs.washington.edu/~todorov/papers/TodorovNIPS06.pdf)
- [Moving Least-squares Approximations for Linearly-solvable MDP](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5967383)
- [Aggregation Methods for Lineary-solvable Markov Decision Process](https://homes.cs.washington.edu/~todorov/papers/ZhongIFAC11.pdf)
- [A Unifying Framework for Linearly Solvable Control](https://arxiv.org/abs/1202.3715)
- [A Stability Result for Linear Markov Decision Processes](http://www.optimization-online.org/DB_FILE/2017/03/5893.pdf)
