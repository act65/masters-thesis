From the data we recieve. Trajectories. How can we infer symmetries in the environment / value / ...

> What is the general formulation of finding symmetries?

Find $f$ s.t. $g(f(x)) = g(x)$.
- $x$ could be the state, an action, a reward, ...
-

***

Find a function $f$ that transforms the state in a way that preserves the change in state of each action.

$$
\mathop{\text{argmin}}_{f} \mathop{E}_{(s, a)}^{(\hat s, \hat a)} \mathbf 1_{a = \hat a}\parallel (\tau(s, a) - s) - (\tau(f(\hat s), a) - f(\hat s)) \parallel_2^2 \\
\mathop{\text{argmin}}_{f} \mathop{E}_{(s, a, s')}^{(\hat s, \hat a, \hat s')}\mathbf 1_{a = \hat a}\parallel (s' - s) - (f(\hat s') - f(\hat s)) \parallel_2^2 \\
$$

Many there are many possible $f$s. Could learn an orthogonal ensemble of them?

***

Find a function $f$ that transform the state in a way that preserves the change in state of actions paired by $g$.

$$
???
$$


***

Find a transformation that

$$
T(Q)(s, a) = r(s, a) +  \gamma \mathop{E}_{s' \sim P(\cdot|s, a)} \mathop{\text{max}}_{a'} Q(s', a') \\
T(Q)(s, a) - Q(s, a) = \delta(s, a) \\
$$

If $\delta(s, a) \approx \delta(s', a')$ what does that tell us!?
What if we had $f, g$ such that $\delta(s, a) \approx \delta(f(s), g(a))$?


***
Want a representation that captures the symmetries / invariances in the environment.

- [A Theoretical Analysis of Contrastive Unsupervised Representation Learning](https://arxiv.org/abs/1902.09229)
- [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)
- [Probabilistic symmetry and invariant neural networks](https://arxiv.org/abs/1901.06082)

Representation learning for RL

- [A Geometric Perspective on Optimal Representations for Reinforcement Learning](https://arxiv.org/abs/1901.11530)
- []()


***
[On Variational Bounds of Mutual Information](https://arxiv.org/abs/1905.06922)
[Mutual information neural estimation](https://arxiv.org/abs/1801.04062)
[On mutual information maximisation for representation learning](https://arxiv.org/abs/1907.13625)
Mutual information doesnt work because?!?
Do some minimal tests to show MI is not what we want!?
