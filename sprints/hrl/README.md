
Define HRL!?

Temoral abstractions of actions.
Actions coordinated over long time periods.
Credit assignment over long time periods.
Exploration in 'meaningful' directions.
Ok, so we wany a multiscale representation?
But also, a multiscale way to pick actions and assign credit.
And, a decompoition of ...?
Understanding how actions combine (this is necessary knowledge for HRL?)


#### Keys to HRL?

- Temporally abstracted actions -- (via options and Goal conditioned policies?)
- Long term credit assignment
- Heirarchical state representations
- Are subgoals necessary for HRL?

## Possible projects

- [ ] Unsupervised options
- [ ] Equivalence of goal/option conditioned value fns
- [ ] Build a three (or even better, N) layer heirarchy
- [ ] Explore how different approaches scale (computational complexity) in the number of heirarchies
- [ ] Using a reachability metric to measure proximity to subgoal (and thus use to give rewards)
- [ ] ?

## Questions and thoughts

- Does it make sense to ask if actions can also be abstracted over dimensions other than time!?
- To learn action abstractions they must capture info about the model. How much harder is it to learn action abstractions in model-free vs model-based settings?

## Refs

- [HLMDPs](https://arxiv.org/abs/1612.02757)
- [Option-critics](https://arxiv.org/abs/1609.05140) and [Abstract options](http://papers.nips.cc/paper/8243-learning-abstract-options.pdf)
- [Modulated policy heirarchies](https://arxiv.org/abs/1812.00025)
- [Feudal](https://arxiv.org/abs/1703.01161) and [HIRO](https://arxiv.org/abs/1805.08296)
- [Model free representations for HRL](https://arxiv.org/abs/1810.10096)
