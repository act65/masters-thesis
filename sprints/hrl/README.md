> 2. Meta-RL [@Wang2017LearningTR] trains a learner on the aggregated return over many episodes (a larger time scale than typical). If we construct a temporal decomposition (moving averages at different time-scales) of rewards and approximate them with a set of value functions, does this naturally produce a rich set of options (/hierarchical RL)?

- motivate the idea as a solution to an existing problem,
!?
- prove that the "existing" problem really exists,
!?

Define HRL!?

Temoral abstractions of actions.
Actions coordinated over long time periods.
Credit assignment over long time periods.
Exploration in 'meaningful' directions.
Ok, so we wany a multiscale representation?
But also, a multiscale way to pick actions and assign credit.
And, a decompoition of ...?
Understanding how actions combine (this is necessary knowledge for HRL?)


Different speeds of learning yield meta learning!? Fast-slow weights. Memory. Worker-manager. ...? Predicting weights.

May as well call reptile/MAML fancy averaging...

Key is credit assignment over many time periods. How to build this!?!?

But if each layer is exploring to estimate its own value and optimal policy. Then how can higher layer account for the extra variance!?

#### Ensemble of critics. Value decomposition (in temporal scale)

Each receiving different inputs?
Or could use fourier TD to estimate. Then we can reover an FFT!?
But what else can it represent? What can vanilla TD not represent? (oscillations!?)

Relationship to something like Rudder!?

#### Multiscale state representation

If we had a multiscale state representation then we could build the policy as a fn of this representation.
Thus adding noise to the higher freq states would result in more local exploration (closer to random?!) and adding noise to the lower freq states would result in 'gobal' exploration over longer time periods!?

(huh, feels weird this has nothing to do with a heirarchical representation of the rewards)

#### Making interventions at various timescales.

There exist N different scales that we can apply interventions at. We want to know;
- what these interventions do
- which interventions lead to the highest reward

What does it mean to be at a different time scale? We get access to subsampled info, or it is averaged or ...!?
Or low/high pass filters? Or !?.



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
- [ ] ?

## Questions and thoughts

- Does it make sense to ask if actions can also be abstracted over dimensions other than time!?
- Relationship between learning to learn and HRL?
- To learn action abstractions they must capture info about the model. How much harder is it to learn action abstractions in model-free vs model-based settings?

## Refs

- [HLMDPs](https://arxiv.org/abs/1612.02757)
- [Option-critics](https://arxiv.org/abs/1609.05140) and [Abstract options](http://papers.nips.cc/paper/8243-learning-abstract-options.pdf)
- [Modulated policy heirarchies](https://arxiv.org/abs/1812.00025)
- [Feudal](https://arxiv.org/abs/1703.01161) and [HIRO](https://arxiv.org/abs/1805.08296)
- [Meta learning overview](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html)
- [Model free representations for HRL](https://arxiv.org/abs/1810.10096)
