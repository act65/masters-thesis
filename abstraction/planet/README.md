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
