%critics
# Critics

## REINFORCE



$$
\begin{align*}
L(\pi) &= \mathbb E_{s\sim\pi}[R(s)] \\
\pi^* &= \mathop{\text{argmax}}_{\pi}  L(\pi) \\
\nabla L(\pi) &= \nabla \mathbb E_{s\sim\pi}[R(s)] \\
&= \nabla \int \pi(s) R(s) \\
&= -\nabla log(\pi(s)) R \\
\end{align*}
$$


(why does less variance mean faster learning!? need to motivate)
Advantage actor critic improves this by reducing the variance of the gradient estimation.

$$
\begin{align*}
A &= R(s) - V(s) \\
&\approx V(s_t) + r(s) - V(s) \\
&= -\nabla log(\pi(s)) A \\
\end{align*}
$$

But what if $V(s)$ is not a reliable estimate of $R(s)$? Are there cases where this could actually give worse behaviour? How about the average case in training?

Hypothesis: because we are using a neural network to estimate $V(s)$, when $r(s)$ is sufficiently sparse, then the neural net will collapse to a near constant function.
Meaning it provides little variance reduction.

## Resources

- [Distributional RL](https://arxiv.org/abs/1806.06923)
- [Backprop through the void](https://arxiv.org/abs/1711.00123)
- [REINFORCE](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
- [A2C](https://arxiv.org/abs/1602.01783)
