Inspired by [Characterising divergence in DQL](https://arxiv.org/abs/1903.08894).

Value iteration + taylor approximation.
- How do the params change via the updates?
- How do changes in the params change the value estimates.
- Combine.

$$
\begin{align}
Q_{t+1}(s,a) &= Q_t(s,a) + \alpha_t (T^{* }Q_t(s,a) - Q_t(s,a)) \\
\theta_{t+1} &= \theta + \alpha_t(T^{* }Q_{\theta_t}(s,a) - Q_{\theta_t}(s,a))\nabla_{\theta}Q_{\theta_t}(s,a) \\
\\
Q_{\theta'}(s,a) &= Q_{\theta}(s,a) + \nabla_{\theta}Q_{\theta}(s,a)^T(\theta' - \theta) + \mathcal O(\parallel \theta' - \theta \parallel^2) \\
\\
Q_{\theta_{t+1}}(s,a) &= Q_{\theta}(s,a) + \nabla_{\theta}Q_{\theta}(s,a)^T\Big(\alpha_t(T^{* }Q_{\theta_t}(s,a) - Q_{\theta_t}(s,a))\nabla_{\theta}Q_{\theta_t}(s,a) \Big) + \mathcal O(\parallel \theta_{t+1} - \theta \parallel^2) \\
Q_{\theta_{t+1}} &= Q_\theta + \alpha K_{\theta}D_{\rho}(T^{* }Q_\theta-Q_\theta) +
\end{align}
$$



What if we set $D_{\rho} = \text{diag}((1-\gamma) \cdot (T-\gamma P_{\pi'}) \cdot d_0)$?
Where using $\pi'$ represents some off policy training. And $\pi = \pi'$ is on-policy?

What about momentum?

$$
\begin{align}
g_t &= (T^{* }Q_{\theta_t}(s,a) - Q_{\theta_t}(s,a))\nabla_{\theta}Q_{\theta_t}(s,a) \\
m_{t+1} &= \beta m_t + g_t \\
\theta_{t+1} &= \theta + \alpha_t m_t \\
\end{align}
$$



What about for policy iteration / policy gradients?


$$
\begin{align}
\theta_{t+1} &= \theta_t + \alpha \mathop{\mathbb E}_{\xi \sim d^{\pi}}  [V(\xi)\sum_{t=0}^T\nabla_{\theta}\log \pi_{\theta}(a_t|s_t)] \\
\pi_{\theta'}(s) &= \pi_{\theta} + \nabla_{\theta} \pi_\theta(s)^T (\theta' - \theta) + \mathcal O(\parallel \theta' - \theta \parallel^2) \\
\pi_{\theta_{t+1}}(s) &= \pi_{\theta} + \nabla_{\theta} \pi_\theta(s)^T \Big(\alpha \mathop{\mathbb E}_{\xi \sim d^{\pi}}  [V(\xi)\sum_{t=0}^T\nabla_{\theta}\log \pi_{\theta}(a_t|s_t)]\Big) + \mathcal O(\parallel \theta_{t+1} - \theta \parallel^2) \\
\pi_{\theta_{t+1}} &=
\end{align}
$$

Hmm. Is this going to be constrained to a normalised distribution?



***


Want;

- on / off policy plots
- Plot $Q = Q + \alpha KD(TQ-Q)$ versus VI (how good / bad is this approximation?)
- Plot $\theta' = \theta + \alpha\nabla Q(TQ-Q)$ versus $\theta' = \theta + \alpha\nabla QK^{-1}(TQ-Q)$ (the corrected version)
- test with / without momentum
- test with overparameterisation
