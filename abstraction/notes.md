Why should we care about HRL?

## Define HRL

Temoral abstractions of actions.(how does this related to a decomposition of rewards)
Ok, so we wany a multiscale representation?
Understanding how actions combine (this is necessary knowledge for HRL?)

__!!!__ The problem we are trying to solve?

> Want to scale to very long sequences. 10 chores, 100 tasks, 1000 movements, 10000 actions...
> Variance in long sequences is a bigger problem than in shorter sequences!?  -- !? random noise slowly erodes the effetos of an action making credit assignment hard!? (could email Emma Brunskill to get more details!?)


Reasons to do HRL??? (want to verify these claims - and have refs for them)

- credit assignment over long time periods (learning faster in one env)
- exploration
- transfer

> Approximation perspective: we have a set of options and we want to use them to approximate the optimal policy. A good set of options can efficiently achieve an accurate approximation.

## Questions and thoughts

- To learn action abstractions they must capture info about the model. How much harder is it to learn action abstractions in model-free vs model-based settings?
- Reward as a function of a subspace of the state space. (this is important for learning abstract representations and actions!?)
- What do cts linear heirarchical actions look like!? and their loss surface!?

## Refs

- [HLMDPs](https://arxiv.org/abs/1612.02757)
- [Modulated policy heirarchies](https://arxiv.org/abs/1812.00025)
- [Model free representations for HRL](https://arxiv.org/abs/1810.10096)


***


## Heirarchical models/transitions
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


## Temporal credit assignment

How does HRL help temporal credit assignment?

- [Temporal credit assignment in RL - Sutton 1984](https://scholarworks.umass.edu/dissertations/AAI8410337/)
- [Hierarchical visuomotor control of humanoids](https://arxiv.org/abs/1811.09656)

Why is it easier to learn in a more temporally abstract space?


> 1) [Temporal abstraction] speeds reward propagation throughout the task.
> 2) Even if the abstract actions are useful, they can increase the complexity of the problem by expanding the action space
> 3) Advantages are dependent of the quality of the abstract actions; ... unhelpful actions, ..., can actually make the problem more difficult for the agent.

 [A neural model of HRL](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0180234)


Credit assignement is harder when there are more possible assignments of credit. (but we are not donig credit assignment!?)


## Simple surface

What does the surface of the simplest RL problem look like!?

http://karpathy.github.io/2016/05/31/rl/

__Q:__ How does the surface change w.r.t actions versus params? How are they related?


#### MNIST RL

Must generate an image. Then we get to evaluate it.
Want to generate 0-9.

How can HRL help?

Actions at pixel level seem wasteful... If I colour this pixel do I get rewarded?

$\tau (s, a) = a, \;\; r(s) = s \cdot w$

Could then go to two actions and see how that changes things!?



***

So if two objects have the same value in a dimension of a disentangled space, we could use that to generalise? They are similar in this restricted / abstract way. Therefore ... ???



***


- Abstraction for evluation versus (greedy) improvement?
- Relationship between tree search and HRL? (divide and conquer for MPC) Recursive subgoal decomposition.  https://arxiv.org/pdf/1706.05825.pdf
- Absolute versus relative perspectives (relationship to subgoals and options)


### Generating temorally abstract action spaces

Pieter Abbeel mentioned an env they designed.
> Grid world where agent must discover the passcodes for the actions (left right up down). For example left might be 0,0,1.
