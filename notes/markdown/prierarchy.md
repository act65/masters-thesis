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
