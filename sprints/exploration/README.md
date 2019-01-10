While learning a model $s_{t+1} = \tau(s_t, a_t)$ is useful. It is more useful to know how to get around using that model. For example, I want to get to $s^k$, how can I do that considering I am in another state, $s^i$? Want a function $f(s^i, s^k) \to \{a_1, a_2, \dots, a_n\}$ that outputs a sequence of actions. You need to know how to get around...

Resources

- [HER](https://arxiv.org/abs/1707.01495)
- [Reachability](https://arxiv.org/abs/1810.02274)
- [Topographic memory](https://arxiv.org/abs/1803.00653)


Options -> model free RL. !?
Reachability -> model based RL!? (distill model into a goal conditioned planner)
Unsupervised options -> !?


https://arxiv.org/pdf/1811.07819.pdf !?


Reachability

Pick two random locations. Start, goal, $s_i, s_g$ and generate a trajectory/policy.

$$
(a_0, a_1, \dots, a_n) \sim \xi(s_i, s_g)  \\
a_i = \pi (s_t, s_g) \\
$$

Trained on !? Proximity to $s_g$? (but how can we measure that!?)

How to learn this?
- Possible to use the model to discover/learn options!?!
- Or gather data in the real world.


What about $f(s_i, s_g) \in [0, 1]$ so that if $s_g$ is reachable from $s_i$ then is 1 else 0. TDMs!?



Relationship to knowledge base completion. Or link prediction!?


Key to exploration is memory.
But what sort of memory?
Explicit memory of all visited states would be sufficient but unnecessary.
In the 1D case it should be easy to see that only knowledge of the boundary is necessary. Aka 2 points. Therefore in higher dims!? $2^d$ points? no. more.


## Controllability

https://en.wikipedia.org/wiki/Controllability




!!! the k steps reachability allows us to construct a metric! thus we can compare states in a meaningful way!?
problem with sub goal approaches was that they relied on having a metric to track progress toward the goal.
so the policy needs to learn to pick actions that will reduce the metric.


PROBLEM:
When modelling/exploring cts state space (with cts actions). How much resolution should we allocate, and where to? Normally RL uses the reward signal to guide the allocation of resources, but in the absence of reward, what should be done?

We could use controllability/observability to help guide the allocation of resources?
If our actions (cts) cause a change in state, but there is noise of var=0.1, then there is no point attempting to build finer grained models!? If we cannot reliably distinguish the results fo two similar but different actions, then treat them as the same.

^^^ What about if there was a cost per action!?

## Stochastic exploration vs deterministic exploration

Want a deterministic exploration policy for bandits!?
Can imagine testing each arm k times, as a fn of its variance.
Iterate through each arm until satisfied.

But this isnt like online learning.
Here there would be two phases? Explore, then exploit.


## Exploration

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

- To build an IID dataset? (important for offline learning)
- To sample states with max entropy (important for online learning)
- In the case when you can never build a large enough/representative dataset, to always find new things!
-

***

PROBLEM:
Might have a highly predictable transition (like a chain, left goes left, ...).
That eventually leads to a reward. Surprise would not find this!?


#### transition fn

Want to construct a dataset where each possible state/transition/reward has uniform probability (aka max entropy)

#### policy

Want to construct a dataset where ...?


#### How do you encourage exploration?

- Novelty, anomalies, information gain, ... Review these!?
