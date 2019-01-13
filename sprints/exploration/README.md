# Exploration

Approaches:
- count based
- intrinsic reward via curiosity/prediction error/novelty
- rnd actions
- ?


What is it for?

- to learn a transition fn
- to evaluate a policy
- to collect data for training
- discovering alternative/better sources of reward. estimating differences.
- finding the solution (like solving a maze or puzzle...)
- to find things not already known

?

- To build an IID dataset? (important for offline learning)
- To sample states with max entropy (important for online learning)
- In the case when you can never build a large enough/representative dataset, to always find new things!
-


### Stochastic exploration vs deterministic exploration

Want a deterministic exploration policy for bandits!?
Can imagine testing each arm k times, as a fn of its variance.
Iterate through each arm until satisfied.

But this isnt like online learning.
Here there would be two phases? Explore, then exploit.


### Memory

Key to exploration is memory.
But what sort of memory?
Explicit memory of all visited states would be sufficient but unnecessary.
In the 1D case it should be easy to see that only knowledge of the boundary is necessary. Aka 2 points. Therefore in higher dims!? $2^d$ points? no. more.



## Alternative views

### Online anomaly detection

A agent gets fed new states (the states reachable from the current state), every time step.
Must determine if they have been seen before. Aka anomaly detection?
If novel then choose relevant action to achieve the novel state.

### Interploation and extraplation

Have seen many data points, and have constructed a map.
Fill in the blank spots.

Closely related to reachability!?

Reachability for exploration. Could learn (from past states), $\pi(s_i, s_g)$. And then use this to extrapolate to reach a new goal. $\pi(s_i, s_a + a_b)$ or $\pi(s_i, s_a + s_i)$ or ...!?

### Unstable dynamical system

Want a maximaly unstable system? One that goes through every state as fast and uniformly as possible?
Orthogonal behaviour to any stable points!?

Is there a notion of maximal instability? Chaos?

### Maximum entropy

Want to construct a dataset where each possible state/transition/reward has uniform probability (aka max entropy)


## Dimensionality reduction (reduce search space)

Exploring in action-state space is a expensive problem. There are many action-states.
We want to abstract into a smaller domain. But how can we make this space smaller?

- factor redundant paths (maybe there are many ways to get from A to B. pick one) (note afawk one path might be no better than others. need something like energy to help make the choice?). aka search in state space, not trajectory space?!
- recover a disentangled representation (if disentangled can explore in parallel!?)
- factor symmetries (maybe search space is large, but after a set of initial actions, the ... or there exists a transformation $t(\pi(s))$ such that ...???) (and approximate symmetries?!)
- only consider exploring controllable action-states.
- ?

### Controllability

https://en.wikipedia.org/wiki/Controllability


!!! the k steps reachability allows us to construct a metric! thus we can compare states in a meaningful way!?
problem with sub goal approaches was that they relied on having a metric to track progress toward the goal.
so the policy needs to learn to pick actions that will reduce the metric.


PROBLEM:
When modelling/exploring cts state space (with cts actions). How much resolution should we allocate, and where to? Normally RL uses the reward signal to guide the allocation of resources, but in the absence of reward, what should be done?

We could use controllability/observability to help guide the allocation of resources?
If our actions (cts) cause a change in state, but there is noise of var=0.1, then there is no point attempting to build finer grained models!? If we cannot reliably distinguish the results fo two similar but different actions, then treat them as the same.


[Actionable representations](https://arxiv.org/abs/1811.07819)

### Symmetries

> Searching based on structure first

Lower bound for searching a 2d graph is $n^2$. We would still need to search all the squares.
But maybe this isnt true. The problems we face are not uniformly distributed.
We would search the 2d space by checking edge cases, and the center, and lines of symmetry.

PROBLEM:
Might have a highly predictable transition (like a chain, left goes left, ...).
That eventually leads to a reward. Surprise would not find this!?


## Effects

> 6. When attempting to learn a model, the agent uses an exploration policy. This policy may influence the dynamics observed, thus we need to use off-policy methods to correct for the effects of exploration actions. (The model must somehow disentangle the agents policy, and its effects, from the dynamics of the system)

Is this actually a problem!? If we are exploring properly the effects should average out? Aka there shouldnt be any bias we need to correct?

Off policy learning!!

Bull in china shop problem. Any exploration will probably break china. How can we disentangle the observations of broken china and ???.

## Questions


- Partial observability. How does this make explortion harder? I have seen X before. (but you might be in a different state). I havent seen Y before (well you were just looking in a different direction last time it occured).
- ?

## Puzzles

### Mis-direction

New search puzzle.
n different directions. All directions have an exploration bonus (crumbs along the way).
It should be easy to learn to explore to the end of each direction?
But for each dimension the agent could detach from the exploration bonus and stop exploring.
What if each dimension each had a different (but constant per dim) exploration bonus. The agent should generalise (quickly) and explore the highest!?

If one direction $= x^2$ and another = $x^3$ then

### external aids

> What if we are able to alter our environment?

Allowed to leave markers (bread crumbs)?
To impose order? (there exists an ordering of the particles, but we dont know what it is)
To get a birds eye view?

How do these make exploration an easier problem?



***

If we imagine a hunter/gather trying to survive in a hostile environment.
Then how do exploration strategies/memory emerge as a necessary criteria for success?


### Resources

- [Disentangled curiosity](https://arxiv.org/abs/1807.01521)
- [Count based exploration](https://arxiv.org/abs/1606.01868)
- [Curiosity](https://pathak22.github.io/large-scale-curiosity/)
