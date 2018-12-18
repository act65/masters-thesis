The equivalence of options and subgoals. Both specify likely future events and allow us to construct temporally abstract actions. Can we show an equvalence between these approaches?

In one case we condition the policy on a subgoal, and in the other case we condition with an option.

Based on [Feudal nets](https://arxiv.org/abs/1703.01161) and [The option-critic architecture](https://arxiv.org/abs/1609.05140)

## Options

$$
\begin{align}
Q_{{\Omega}}(s, w) &= E_{a\sim \pi_{\omega}(s)}[Q_{\omega}(s, w, a)] \tag{over multiple time steps!??!}\\
Q_{\omega}(s, w, a) &= r(s_t) + \gamma  E_{s' \sim \tau(s, \pi(s))}[U(s', w)] \\
U(s, w) &= (1-\beta(s, w))Q_{\Omega}(s, w) + \beta(s,w) V_{\Omega}(s)
\end{align}
$$

(these equations are copied from the OpC paper.)

## Subgoals

- $Q_{\Omega}(s, g)$ is the expected discounted reward of choosing using $\pi_{\omega}$ to reach the subgoal, $g$ and then following $\pi_{\Omega}$ afterwards.
- $Q_{\omega}(s, g, a)$ is the expected discounted reward of choosing an action given that we are attempting to achieve some goal.

= Value of subgoal + value of following subgoal policy in the future.
$$
\begin{align}
Q_{\Omega}(s, g) &= r(s_t) + \gamma  \mathop{E}_{s' \sim \tau(s, \pi_{\omega}(s, g))}[U(s')] \tag{manager}\\
U(s, g) &=  (1-\beta(s, g))Q_{\Omega}(s, g) + \beta(s, g) V^{\pi_{\Omega}}_{\Omega}(s)\\
r_{\omega}(s_t, g) &= \gamma\mathop{E}_{s' \sim \tau(s, \pi_{\omega}(s, g))} [Q_{\Omega}(s', g)] - Q_{\Omega}(s_t, g) \tag{manager rewards the worker}\\
Q_{\omega}(s_t, g_t, a_t) &= r_{\omega}(s_t, g) + \gamma E_{a \sim \pi(s')}[Q_{\omega}(s_t, g_t, a_t)] \tag{worker}\\
\end{align}
$$

(1) This is a non-standard definition. $r_{\omega}(s_t, g) = \mathop{E}_{s' \sim \tau(s, \pi_{\omega}(s, g))} [Q_{\Omega}(s', g)] - Q_{\Omega}(s_t, g)$ But can we justify it? (_intuitively it makes sense that we would reward the worker if it increases the expected rewards!?_) To be able to caluclate this we will approximate it with $r_{\omega}(s_t, g) = Q_{\Omega}(s_{t+1}, g) - Q_{\Omega}(s_t, g)$. As long as $s_{t+1}$ is sampled IID from $\tau(s_t, \pi(s_t, g_t))$ then this estimator should have zero bias (but we have introduce more variance).

(1.1) Also, this because it is more general? Other approaches to subgoals 'cheat'(??) in the sense that they use the euclidean distance between the current state and goal state (see TDMs).

(1.2) This trick is what connects the two Q fns. As now the both have the same reward fn. (see section ...)

(2) The introduction of the $\beta$ is also non-standard for the Feudal net framework. But can we justify it? Feudal networks are implemented by running the manager at a lower temporal resolution than the worker. Thus the worker may recieve the same goal for $k$ steps. We can use $\beta$ to index the correct discounted reward. Maybe the worker policy is simply following the current subgoal, and thus the expected discounted reward is the value of the workers policy for the next k steps, in which case $\beta$ should be zero for these k steps. Else, for $\beta$ equals one, the manager picks a new subgoal, in which case we can recursively define it value as the value of the rolledout workers actions.

***

$V_{\omega}(s) = \sum_t \gamma^t r_w(s) = \sum_t \gamma^t r(s)$ is the ideal case!? The problem is that it is hard to estimave the correct value of an action, especially with long term deps and sparse rewards. Thus we choose $r_w(s)$ to ...?

## Equivalence

$$
\begin{align}
Q_{\omega}(s, w, a) &= \sum_t \gamma^tr(s_t) \\
&= Q_{\omega}(s, g, a) \\
\end{align}
$$

#### Generalisation/calc interpretation

> Roughly. Option-critics take the integral of $Q_{\omega}$ to construct $Q_{\Omega}$ while feudal networks take the derivative of $Q_{\Omega}$ to construct $Q_{\omega}$ _(assuming the introduction of 1, 2)_.



$$
\begin{align}
&\text{definition of option manager}\\
Q_{{\Omega}}(s, w) &= \int p(a|s, \pi_{\omega})Q_{\omega}(s, w, a) da
\end{align}
$$

So what is this reward, $r_w$? A way to think about it could be as the ...!?


(3) $a_{t}$ is via greedy policy
(4) at convergence og Q to the true value of the current policy

$$
\begin{align}
r_{\omega}(s_t, g) &= \gamma Q_{\Omega}(s_{t+1}, g) - Q_{\Omega}(s_t, g) \tag{by defn} \\
&= r(s_t) \tag{assuming (3), (4)}\\
 Q_{\Omega}(s_{t+1}, g) - Q_{\Omega}(s_t, g) &\approx \frac{\partial Q_{\Omega}}{\partial t} \\
Q_{\omega}(s, g, a) &= E_{}[\gamma^t r(s_t)]\\
&= \int \gamma^t r(s_t) dt \\
\int Q_{\omega}(s, g, a) da &=
\end{align}
$$

***

Ok. I am starting to have second thoughts. Not sure if this is insightful or trivial... I mean I just defined the manager $\Omega$ and the worker $\omega$ to recieve the same reward. And showed that integrating the rewards gives a value fn and differentiating the value fn gives the reward... Doesnt seem particularly interesting...

Oh... I think I am wrong.

## Comparison

pros/cons

- (con - options) calculating an integral for every choice...
- (con - subgoals) manager must be accurate, else can introduce bias/variance
- ?

***

#### Scaling HRL

For __tabular FuN__ we have;
- `manager_qs=[n_states x n_states] (current_states x goal_states)` and
- `worker_qs=[n_states x n_states x n_actions] (current_states x goal_states x actions)`.

But what if we wanted to increase the depth of the heirarchy?
- `Exectuive_manager_qs=[n_states x n_states] (current_states x exec_goal_states)` and
- `manager_qs=[n_states x n_states x n_states] (current_states x exec_goal_states x goal_states)` and
- `worker_qs=[n_states x n_states x n_actions] (current_states x goal_states x actions)`.

For every added layer of depth, the increase complexity is upperbounded by (d = depth) $d \times n_{subgoals}^3$ (where $n_{subgoals} = n_{states}$) (?!?!)

(__^^^__ reminds me of some sort of tensor factorisation!? __!!!__)

But for __tabular OpC__. For tabular FuN we have;
- `qs=[n_states x n_options x actions] (current_states x options x actions)`

But what if we wanted to increase the depth of the heirarchy?
- `qs=[n_states x n_options x n_options x actions] (current_states x 1st_lvl_options x 2nd_lvl_options x actions)`
- and we would need to compute integrals of each layer of depth (so memory and time complexity are bad)

The increase is upperbounded by $n_{options}^{d}$. (?!?)

__BUT.__ What about scaling with state size?

- OpC - $O(n)$
- FuN - $O(n^3)$

***

Why would we want more depth? (greater length of time deps?? which is related to state size!?)


## TODOs

- Is there a nice way to visualise this!? Pictures!?
- ?
