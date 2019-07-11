Blah blah. [Challenges of Real-World Reinforcement Learning](https://arxiv.org/abs/1904.12901).
- Learning on the real system from limited samples.
- High-dimensional continuous state and action spaces.


## Reinforcement learning

> Reinforcement learning (RL) defines a type of problem, closely related to Markov decision problems (MDPs).

A Markov decision problem is defined as the tuple, $\{\mathcal S, \mathcal A, P, r\}$. Where $s \in \mathcal S$ is the set of possible states (_for example arrangements of chess pieces_), $a \in \mathcal A$ is the set of actions (_the different possible moves, left, right, diagonal, weird L-shaped thing, ..._),  $P: \mathcal S\times \mathcal A \times \mathcal S \to [0:1]$ is the transition function which describes how the environment acts in response to the past ($s_t$) and to your actions ($a_t$) (_in this case, your opponent's moves, taking one of your pieces, and the results of your actions_), and finally, $r: \mathcal S\times \mathcal A \to \mathbb R$ is the reward function, (_whether you won (+1) or lost (-1) the game_) and $R = \sum_{t=0}^T \gamma^t r(s_t, a_t)$ is the discounted cumulative reward, or return. The player's goal, is to find a policy $\pi$, (which chooses actions, $a_t = \pi(s_t)$) that yields the largest return ($\text{max } R$).

A RL problem is an extension of the MDP definition adove. Where, rather than the learner being provided the state space, action space, transition function and reward function ($\{\mathcal S, \mathcal A, P,r\}$), the learner recieves samples $(s_t, a_t, r_t)$. From these samples the learner can either;
- attempt to infer the transition and reward functions (known as model-based reinforcement learning), or attempt to estimate value directly (model-free reinforcement learning).
- collect the samples in memory and use them to find a policy (offline learning), or
- on / off policy
- bootstrap / not
- types of model (fn approximators)

For example _"Dynamic programming is one type of RL. More specifically, it is a value-based, model-based, bootstrapping and off-policy algorithm. All of those traits can vary. Probably the "opposite" of DP is REINFORCE which is policy-gradient, model-free, does not bootstrap, and is on-policy. Both DP and REINFORCE methods are considered to be Reinforcement Learning methods."_ [SE](https://datascience.stackexchange.com/questions/38845/what-is-the-relationship-between-mdp-and-rl)

### Understanding Theoretical Reinforcement learning

What are its goals. Its definitions. It methods?

- Optimality
- Model based
- Complexity
- Abstraction

Recent work has bounded the error of representation learning for RL. [Abel et al. 2017](), [Abel et al. 2019]()

But. It is possible that this representation achieves no compression of the state space, making the statement rather vacuous.
Further more, it consider how easy it is to find the optimal policy in each of the two representations. It is possible to learn a representation that makes the optimal control problem harder. For example. TODO



Current theory does not take into account the structure within a RL problem.

The bounds are typically for the worst case.
But these bounds could be tighter if we exploited the structure tht exists in natural problems.
The topology of the transition function; its, sparsity, low rankness, locality,
The symmetries of the reward function.
??? (what about both?!)

### Understanding Markov decision problems

- Properties of the polytope
- Search dynamics on the polytope
- LMDPs

## Overview

We explore four algorithms.

- Memorizer: This learner memorizes everything it sees, and uses this knowledge as an expensive oracle to train a policy.
- Invariant. This learner discovers symmetries in its evironment and uses this knowledge to design an invariant representation.
- Tabular. ...
- MPC. ...
