__Claim:__ Low policy entropy implies low evaluation variance.

> If the prior is low entropy, then you should be able to significantly increase the discount factor. This is because the noise of policy gradient algorithms scales with the amount of information that is injected into the trajectories by sampling from the policy. [ref](https://blog.aqnichol.com/2019/04/03/prierarchy-implicit-hierarchies/)

$$
\begin{align}
H(\pi(\cdot | s)) &=  \mathop{\mathbb E}_{a \sim \pi(\cdot | s)} -\ln \pi(a | s)\\
H(\pi) &= \mathop{\mathbb E}_{s \sim D_{\pi}} H(\pi(\cdot | s)) \\\\
J(\pi) &= \mathop{\mathbb E}_{s \sim D_{\pi}}[V^{\pi}(s)] \\
\frac{\partial J}{\partial \pi} &= \\
\text{var}(\frac{\partial J}{\partial \pi}) &= \mathop{\mathbb E} [(\frac{\partial J}{\partial \pi} - \mathop{\mathbb E} \frac{\partial J}{\partial \pi})^2]
\end{align}
$$


or variance in the gradient estimate!?!?
is variance in J prop to var in dJ?


$$
\begin{align}
\nabla_{\pi} J(\pi) &= \nabla_{\pi}\mathop{\mathbb E}_{s \sim D_{\pi}}[V^{\pi}(s)] \\
&= \nabla_{\pi}\int D_{\pi}(s) V^{\pi}(s)ds \\
&=\int \nabla_{\pi} D_{\pi}(s) V^{\pi}(s)ds \\
\end{align}
$$
