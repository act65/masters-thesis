The whole point of abstraction is that it makes the problem easier to solve.
So, lets explore some

Search (MPC/MCTS/?) + abstraction.
https://github.com/google-research/planet

Min one of:
* [ ] memoize
* [ ] multi-scale transitions ([HYPERBOLIC DISCOUNTING AND LEARNING OVER MULTIPLE HORIZONS](https://arxiv.org/pdf/1902.06865.pdf), separating value fns across time)
* [x] disentangled actions
(maybe more depending on how easy code/experiments are)

***

- if latent space is disentangled. order of actions may not matter, bc the mechanisms are independent!? could run a single rollout in parallel then!?
- also. how does having disentangled action aid planning?

***

Improvement to planet.
Learn the reward fn as its stationary. Value can then be estimated based on the current policy!?

***

What env/task/problem?

Just some of the retro games?!


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
