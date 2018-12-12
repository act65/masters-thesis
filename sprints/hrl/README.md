> 2. Meta-RL [@Wang2017LearningTR] trains a learner on the aggregated return over many episodes (a larger time scale than typical). If we construct a temporal decomposition (moving averages at different time-scales) of rewards and approximate them with a set of value functions, does this naturally produce a rich set of options (/hierarchical RL)?

Define HRL!?

Temoral abstractions of actions.
Actions coordinated over long time periods.
Credit assignment over long time periods.
Exploration in 'meaningful' directions.
Ok, so we wany a multiscale representation?
But also, a multiscale way to pick actions and assign credit.
And, a decompoition of ...?

Keys to HRL?

- Goal conditioned policies?
- Options
- ?

Refs

- [HLMDPs](https://arxiv.org/abs/1612.02757)
- [Option-critics](https://arxiv.org/abs/1609.05140) and [Abstract options](http://papers.nips.cc/paper/8243-learning-abstract-options.pdf)
- [Modulated policy heirarchies](https://arxiv.org/abs/1812.00025)
- [Feudal](https://arxiv.org/abs/1703.01161) and [HIRO](https://arxiv.org/abs/1805.08296)(<!)
