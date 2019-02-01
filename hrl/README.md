Problem to solve:

- in what cases does HRL actually help?
- why should we care about HRL?
-


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


#### Keys to HRL?

- Temporally abstracted actions -- (via options and Goal conditioned policies?)
- Long term credit assignment
- __Q:__ Are subgoals necessary for HRL?

## Possible projects

- [ ] Unsupervised options (how good can random ones be!?)
- [ ] Equivalence of goal/option conditioned value fns
- [ ] Build a three (or even better, N) layer heirarchy (!!!)
- [ ] Explore how different approaches scale (computational complexity) in the number of heirarchies
- [ ] Using a reachability metric to measure proximity to subgoal (and thus use to give rewards)
- [ ] A heirarchical subgoal net that uses MPC rather than learned policies
- [ ] A generator for options a = f(w) (rather than look up table)
- [ ] Recursive subgoal decomposition
- [ ] Why doesnt Feudal learning work?? (it does but other researchers have had trouble -Abbeel...?)
- [ ] More complex action spaces!!
- [ ] A way to generate toy HRL puzzles/envs. Want a set of priors! What properties do they have?!
- [ ] What if we could learn a disentangled set of actions!? And then compose them into more abstract actions!? How do disentangled transitions correspond to disentangled actions!? (in what sense would the actions be disentangled. in the change in state?)
- [ ] When is negative transfer guaranteed not to occur!?

> Approximation perspective: we have a set of options and we want to use them to approximate the optimal policy. A good set of options can efficiently achieve an accurate approximation.

## Questions and thoughts

- Does it make sense to ask if actions can also be abstracted over dimensions other than time!?
- To learn action abstractions they must capture info about the model. How much harder is it to learn action abstractions in model-free vs model-based settings?
- Reward as a function of a subspace of the state space. (this is important for learning abstract representations and actions!?)
- What do cts linear heirarchical actions look like!? and their loss surface!?

***


Can have temporal and action abstractions.

There exist a set of disentangled higher level actions that explain the change in state / transition fn.

***

> there is still no consensus on what constitute good options. [A Matrix Splitting Perspective on Planning with Options](https://arxiv.org/pdf/1612.00916.pdf)
> if the option set for the task is not ideal, and cannot express the primitive optimal policy well, shorter options offer more flexibility and can yield a better solution. [Learning with Options that Terminate Off-Policy](https://arxiv.org/pdf/1711.03817.pdf)

## Refs

- [HLMDPs](https://arxiv.org/abs/1612.02757)
- [Modulated policy heirarchies](https://arxiv.org/abs/1812.00025)
- [Model free representations for HRL](https://arxiv.org/abs/1810.10096)
