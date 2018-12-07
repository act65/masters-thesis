What is the difference between exploration and exploration?

A learner needs to explore its environment.
A learner needs to explore alternative policies.

Decoupling these via model-based RL means that when we want to evaluate a policy we do not need to explore/cover the environment. Only the simulation.


Setting.

Receive observations, take actions. Goal is to !?






Possible to learn a model via rewards?
Rewarded for small memory.
Rewarded for accurate predictions.
Rewarded for ability to explain!?



## Rewarded for explanations

The oracle takes a program specification, $P$ and uses it to generate the target observation, $x$.

$$
\mathop{\text{min}} d(x, P()) \text{ s.t } P = G(x)
$$

It seems like this would need a curriculum!?

## Rewarded for understanding

How can this be rewarded!?
Will be a natural result of compression!?

## Rewarded for compression

!?
