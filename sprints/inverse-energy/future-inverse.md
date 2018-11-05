What is the future of inverse learning?

Inverse learning can be summarised as learning a functional that is being optimised.

Classical mechanics can be modelled as minimising the 'action'.
People and learned agents can be modelled as maximising their rewards.
X can be modelled as ???.  

The future of marketing: facebook (/similar) will observe you and infer the experiences you find rewarding.

In nature, we commonly observe the optimal policy under the [principle of least action](https://en.wikipedia.org/wiki/Principle_of_least_action). The lagrangian defines an energy function, say of ... which is minimised.

In economies (or social environments) there are many agents with their own agendas. We tend to take actions that maximise our own rewards.




Big questions

- Is it a universal approximator? (what can we not do with it?)
- In which cases does it work (better)?
- Global view versus local view


### Related work

Relationship to energy-based ML?

Energy-based ML use a fixed measure of energy, for example, [equilibrium propagation](https://arxiv.org/abs/1602.05179):

$$
E(u) = \frac{1}{2}\sum_i u_i^2 - \frac{1}{2} \sum_{i\neq j} W_{ij}\rho(u_i)\rho(u_j) - \sum_i b_i\rho(u_i)\\
$$

And adapt the connectivity/topology of the network. Similarly; hopfield nets, RBMs, ...
