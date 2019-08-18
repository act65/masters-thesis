We have linearised around the optimal policy.
(how can we see this in the math??)
(what do we mean by linearised?)

In what cases does this linearisation not make sense?
Can we try linearising around other points?



$$
\begin{align}
v(s) &= q(s) + \log G(s) + \mathop{\text{min}}_{u} \bigg[\text{KL}\big(u(\cdot | s) \parallel \frac{p(\cdot | s)\cdot z(\cdot)^{\gamma}}{G(s)} \big) \bigg] \\
\dots \\
z_{u^{* }} &= e^{q(s)}\cdot P z_{u^{* }}^{\gamma}\\
\end{align}
$$

Set $\frac{\partial v}{\partial u}(u^{* }) = \mathbf 0$.

Related to a taylor expansion?
