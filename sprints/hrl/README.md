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

Keys to HRL?

- Goal conditioned policies?
- Options
- ?


Can we prove an equivalence between options (policy conditioned value fns) and goal conditioned value fns?

$$
\begin{align}
\omega_i \in \Omega \tag{options}\\
Q(s_t, \omega_t, a) \tag{option conditioned}\\
s_g \in X \tag{goals}\\
Q(s_t, s_g, a) \tag{goal conditioned}\\
\end{align}
$$

What makes a good subgoal or termination point? How are they equivalent?

Should be able to show that they are 'dual' to each other!?

***

What are the pros/cons?
- Top of heirarchy gets reward. (Feudal)
- Bottom of heirarchy gets reward. (options) <-related to GVFs? cumulating upwards?
- All get reward?

## Possible projects

- unsupervised options
- equivalence of goal/option conditioned value fns
- ?

## Refs

- [HLMDPs](https://arxiv.org/abs/1612.02757)
- [Option-critics](https://arxiv.org/abs/1609.05140) and [Abstract options](http://papers.nips.cc/paper/8243-learning-abstract-options.pdf)
- [Modulated policy heirarchies](https://arxiv.org/abs/1812.00025)
- [Feudal](https://arxiv.org/abs/1703.01161) and [HIRO](https://arxiv.org/abs/1805.08296)(<!)
