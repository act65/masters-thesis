When learning the value of one state, we could assume that another state is likely to share that same value. If this assumption is correct, we could increase the speed of learning. If not, the optimisation may become less stable.

When using continuious function approximation of the value function, $V: S \times \theta \to \mathbb R$, such as a neural network,

That is, by chaging a single parameter we change the value for multiple states.

Refs

- [Towards characterising divergence in deep q learning](https://arxiv.org/abs/1903.08894)
- [A Fine-Grained Spectral Perspective on Neural Networks](https://arxiv.org/abs/1907.10599)
