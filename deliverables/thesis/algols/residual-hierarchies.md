

## Heirarchical

- [Hierarchy through Composition with Linearly Solvable Markov Decision Processes](https://arxiv.org/abs/1612.02757)
- [Alex Nichol's preirarchy](https://blog.aqnichol.com/2019/07/24/competing-in-the-obstacle-tower-challenge/)


## Residual

- [Residual Reinforcement Learning for Robot Control](https://arxiv.org/pdf/1812.03201.pdf)
- [Residual Policy Learning](https://arxiv.org/abs/1812.06298)
- [Deep Residual Reinforcement Learning](https://arxiv.org/abs/1905.01072)



and boosting?


Want to show a connection between the prierarchy idea and the composability of LMDPs.

$$
\begin{align}
v(s) &= \mathop{\text{min}}_a \big[ r(s, a) + \mathop{\mathbb E}_{x' \sim p(\cdot | x, a)} v(x') \big]\\
p_i(x' | x) &= \tau(x' | x, \pi_i(x)) \\
r(s, a) &= q(x) +  \mathop{\mathbb E}_{x'\sim a(\cdot | x)} \log \frac{a(x' | x)}{p(x' | x)} - \sum_{i=1}^k \mathop{\mathbb E}_{x'\sim a(\cdot | x)} \log p_i(x' | x) \\
&= q(x) +  \mathop{\mathbb E}_{x'\sim a(\cdot | x)} \log \frac{a(x' | x)}{p(x' | x)\prod_{i=1}^k p_i(x' | x)} \\
\end{align}
$$

These priors maybe added sequential as training progresses.

$$
\begin{align}
-log(x) &= q(x) + \mathop{\text{min}}_a \big[\mathop{\mathbb E}_{x'\sim a(\cdot | x)} \log \frac{a(x' | x)}{z(x')p(x' | x)\prod_{i=1}^k p_i(x' | x)}\big]\\
\end{align}
$$

???

$$
\begin{align}
d(\cdot | x) &= \prod_{i=1}^k p_i(x' | x) \\
a^{* }(x' | x) &= \frac{p(x' | x)z(x')d(\cdot | x)}{G[z](x)} \\
G[z](x) &= \mathop{\mathbb E}_{x'\sim p(\cdot | x)d(\cdot | x)} z(x') \\
&=  \mathop{\mathbb E}_{x'\sim d(\cdot | x)} \mathop{\mathbb E}_{x'\sim p(\cdot | x)}z(x') \\
z_k(x) &:= \mathop{\mathbb E}_{x'\sim p(\cdot | x)p_k(\cdot | x)}z(x') \\
G[z](x) &= \sum_k w_k z_k(x') \\
z(x) &= e^{-q(x)}G[z](x) \\
\end{align}
$$


Want to find a relationship to this.
http://papers.nips.cc/paper/3842-compositionality-of-optimal-control-laws.pdf


Kinda of like auto-completing. Higher level policy picks a couple of initial actions. Then the thought / policy is completed with a lower level policy.


Main thing I am interested in. How to get temporal abstraction via learning heirarchical policies.

Effectively, the $z$s are picking distributions over future states!? Is the an effiient approximation we can use for large state spaces!?


A soft relaxation of choosing which policy to follow via entropy.
$$
z(s) = \sum_i \alpha_i z_i(s) \\
z = \sum_i (1-H(z_i(s))) \cdot z_i(s) \\
z = \sigma(-H(\mathbf z(s)))  \cdot z_i(s) \\
$$

What about adding a penalty to some of the $\alpha$s? Some policies are cheap, some are expensive.

Out of this ensemble, temporal abstraction naturally emerges!?

If we look at the $H(z_i)$, we should see;
- there exist $i$ with low entropy for periods of time
- there should exist $i$ that gets called infrequently?!
- the 'power spectrum' of the policies should be distributed (linearly, exponentially, ... -> not constant.)

***

Learning an ensemble of policies.
- With different lrs?
- with different discounts? (related to https://arxiv.org/abs/1902.01883 and LMDPs)
- with different contextual info?
- ???



***

Intro

There have been many appraoches to heirarchical RL. The motivation being...

Despite these many approaches, there has yet to be a principled approach that actually works.
