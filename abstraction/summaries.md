## Pieter Abbeel talking at NIPs workshop on HRL
(https://www.youtube.com/watch?v=WpSc3D__Av8)

- Information theoretic perspective: if you set a goal now, that you tell you something about the future
- Grid world where agent must discover the passcodes for the actions (left right up down). For example left might be 0,0,1.


> there is still no consensus on what constitute good options. [A Matrix Splitting Perspective on Planning with Options](https://arxiv.org/pdf/1612.00916.pdf)
> if the option set for the task is not ideal, and cannot express the primitive optimal policy well, shorter options offer more flexibility and can yield a better solution. [Learning with Options that Terminate Off-Policy](https://arxiv.org/pdf/1711.03817.pdf)


## A Theory of State Abstraction for Reinforcement Learning
(https://david-abel.github.io/papers/aaai_dc_2019.pdf)

> I propose three desiderata that characterize what it means for an abstraction to be useful for RL:
> 1. SUPPORT EFFICIENT DECISION MAKING: The abstraction enables fast planning and efficient RL.
> 2. PRESERVE SOLUTION QUALITY: Solutions produced from the abstracted model should be useful enough for solving the desired problems.
> 3. EASY TO CONSTRUCT: Creating the abstractions should not require an unrealistic statistical or computational budget.


## On the necessity of abstraction
(https://www.sciencedirect.com/science/article/pii/S2352154618302080)

> The challenge in the single-task case is overcoming the additional cost of discovering the options; this results in a narrow opportunity for performance improvements, but a well-defined objective. In the skill transfer case, the key challenge is predicting the usefulness of a particular option to future tasks, given limited data.

## Near optimal representation learnning for HRL
(https://openreview.net/forum?id=H1emus0qF7)



$$
\begin{align}
D_{KL}(P_{\pi^{* }(s, g)}(s'\mid s) \parallel K_{\theta}(s' \mid s, g)) \\
K_{\theta}(s' \mid s, g) = \rho(s') \frac{e^{-E(s', s, g)}}{Z} \\
E(s', s, g) = D(f(s'), \varphi(s, \psi(s, g))) \tag{distance is measured in abstract space}\\
\end{align}
$$
Kinda like an autoencoder?
$$
\begin{align}
D_{KL}(P(x) \parallel K(x' \mid x)) \\
K(x \mid x) = \rho(x) \frac{e^{-E(x)}}{Z} \\
E(x) = D(x, d(e(x))) \\
\end{align}
$$
