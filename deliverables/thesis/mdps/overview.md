<!--
Tabular
Polytope and properties
Search spaces
Transitions
-->

What is a decision problem?


## Sequential decision problems

Define. And give example.

If we wanted we could pick our actions before we make observations, reducing the search space to only $|A| \times T$. But this is a bad idea... example.

## MDPs

MDPs are a subset of sequential decision problem.
Define MDPs. Give example.

When actions you have taken in the past can bite you in the butt...
Maze with pendulums / doors. When moving through the maze, you must swing the pendulums. In the future you must avoid being hit.
(maybe make a picture of this?)
also, is there a more general way to think about it?


The general feeling of an MDP.
- Actions need to be adapted to new observations and contexts.
- While instantaneous results are good, we care about the longer term aggregates.

### The Markov property

What does the M in MDP mean?

When we say a decision problem is Markovian, we mean that the transition function acts as a Markov chain. The next transition step depends only on the current state. It is invariant to any / all histories that do not change the current state.

This is not to say that past actions do not effect the future. Rather, it is a special type of dependence on the past. Where the dependence is totally described by changes to the __observable__ state.

Can easily make a sequence Markovian by adding information. E.g. time...

### Optimality

And importantly, existing theory tells us that there is a unique optimal policy. And that this optimal policy is necessarily deterministic.

(why does this make sense?)

### How do MDPs relate to RL?

Reinforcement learning set of solutions to a general type of problem. This general, reinforcement learning problem, has the properties;

- evaluation, not feedback. Learners are never provided information about what makes a good policy, rather they told whether a policy is good or not.
- delayed credit assignment.

MDPs have these properties, so are considered within RL. They are also within the fields of Operational Research, Optimal Control, Mathematical Optimisation, Stochastic Programming.

Notably, we can weaken the information provided to a learner attempting to solve an MDP in a few interesting ways:

- POMDPs
- ??? only provided samples from transition an reward.


### A tabular representation of MDPs

Tabular MDPs with deterministic actions are of little interest to the ML community. Not because they are easy, but because they do not involve ...? They can be solved by planning techniques and dynamic programming.

The minimally complex MDP that poses an interesting challenge to the ML community is when the transition function is non deterministic.
Alternatives we could add on. Contextua decision problem (transition fn changes with t), stochastic reward function, ...?

Learn a tabular MDP representation of the RL problem.

Why would we want to do this?
- Policy evaluation is expensive in the RL setting. The policy must be simulated over all possible states-action pairs. And scales poorly with variance. (how poorly?)
- ?

Just quickly, what does a tabular MDP look like?
- discrete states and actions
- $r$ and $P$ are simply look up functions, indexed by the current state-action.

$$
\begin{align}
V &= r_{\pi} + \gamma P_{\pi} V \tag{bellman eqn}\\
V - \gamma P_{\pi} V &= r_{\pi}\\
(I-\gamma P_{\pi})V &= r_{\pi}\\
V &= (I-\gamma P_{\pi})^{-1}r_{\pi}\\
\end{align}
$$

(finding the optimal policy is still a non-linear problem. how / why is it non-linear?!)


### Learning the (tabular) abstraction

Most recent advances in ML have been by finding clever ways to extend supervised learning techniques to unsupervised learning. Similarly, we can use supervised learning techniques, batch training, cross entropy, ... to train reward and transition approximations.

We are provided with examples $(s_t, a_t, r_t, s_{t+1}, a_{t+1})$. We can use these to...

$$
\begin{align}
\textbf  r \in \mathbb R^{n \times m}, &\; \textbf P \in [0,1]^{n \times m \times n} \\
L_{r} &= \text{min} \parallel r_t - \textbf r[\phi(s_t), a_t] \parallel^2_2 \tag{mean squared error}\\
L_{P} &= \mathop{\text{max}}_{\theta} \textbf P[\phi(s_{t+1}),\phi(s_t), a_t]\tag{max likelihood}\\
\end{align}
$$

## MDPs in the real world

Insert some fun worked examples of MDPs from every day life.
- Gwern and catnip / mail / ...

Point out some important applications of MDPs;

- Energy markets. https://www.cem.wi.tum.de/index.php?id=5&L=1
- Medicine (EVI)
- OR



***

Aftrethoughts.

Why is the discount factor a part of the definition of the MDP? Initially, it didnt make sense to me.
By defining the discount, it ensure the MDP has a unique solution.


***

Note: I have not considered MDPs in their many varying dimensions.
Rather than;
- pick the state space, or the action space, to be a discrete set, we could pick any set we like.
- allowing an infinite number of steps, we can add terminal states. This allows us to drop the need for discounting (as the integral will not necessarily diverge to infinity).
-
