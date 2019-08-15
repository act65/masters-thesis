> The whole point of abstraction is to make the problem easier. We throw away the unimportant parts so we can focus on the essential.

Solving a problem via abstraction follows a generic formula. Transform the the problem into a new domain, solve the problem in this new domain, project the solution back into the original domain. In this section, we explore the way to design an abstraction so that it makes the optimisation problem 'easier'.

What makes an optimisation problem easy? The search space could be small, the search space have some structure or symmetry within it, allowing us to reduce the search space to something small(er), the search space gives us hints about how to search for what we are looking for (for example smooth and convex with respect to our loss function [convex RL](https://bodono.github.io/thesis/bod_thesis.pdf) ...), the search space could be 'similar' to a problem we have already solved, ...

And within RL, properties that would make the optimisation problem easier; a sparse transition matrix, dense rewards, linearity, ...

***

Existing work, ... It's even a started technique used by mathematicians (give example)


Give pseudo definition?!

Mention the importance of temporal abstraction.

Relate abstraction to heirarchical RL.

Related work

- https://arxiv.org/abs/1901.11530
- latent space work
- action abstraction?!
- https://www.sciencedirect.com/science/article/pii/S2352154618302080
