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

#### Model-based RL

...

## Representation learning and abstraction

The goal is to find a representation that decomposes knowledge into its parts.

Another way to frame this is: trying to find the basis with the right properties.

- sparsity,
- independence,
- multi scale,
- locality/connectedness
- ???



### Action abstraction

Include my caterpillar here.


## Related work

#### MDPs

Dynamic programming, linear programming, ...?

##### HRL

Temoral abstractions of actions.(how does this related to a decomposition of rewards)
Ok, so we wany a multiscale representation?
Understanding how actions combine (this is necessary knowledge for HRL?)


Reasons to do HRL??? (want to verify these claims - and have refs for them)

- credit assignment over long time periods (learning faster in one env)
- exploration
- transfer

- To learn action abstractions they must capture info about the model. How much harder is it to learn action abstractions in model-free vs model-based settings?
- Reward as a function of a subspace of the state space. (this is important for learning abstract representations and actions!?)
- What do cts linear heirarchical actions look like!? and their loss surface!?

- [HLMDPs](https://arxiv.org/abs/1612.02757)
- [Modulated policy heirarchies](https://arxiv.org/abs/1812.00025)
- [Model free representations for HRL](https://arxiv.org/abs/1810.10096)
- [Prierarchy: Implicit Hierarchies](https://blog.aqnichol.com/2019/04/03/prierarchy-implicit-hierarchies/)


#### Pre training



#### Pre conditioning
