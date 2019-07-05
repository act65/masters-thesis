Planning using a model of the dynamics and a cost function.
Don't want to recalculate decisions.
Want to transfer knowledge between similar decisions.

This is just a dynamic programming problem.
Search plus memoization.


MB-DRL
- The search can be done using MCMC.
- The memoization can be done with a neural network, the policy.

MCTS
- The search can be done using MCMC.
- The memoization can be done with a tree.

CFR
- ?
- Tree


Ultimately we want an incremental way build the optimal policy.
But. We might have approximations of the dynamics model and the cost function.
They might me inaccuate.
We want a representation of the policy than can be efficiently updated given new knowledge of inaccuracy in the model / cost function.


The search needs to be;
- ?


The memory needs to be;
- easily indexed by context (observations)
- ?
