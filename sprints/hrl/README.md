> 2. Meta-RL [@Wang2017LearningTR] trains a learner on the aggregated return over many episodes (a larger time scale than typical). If we construct a temporal decomposition (moving averages at different time-scales) of rewards and approximate them with a set of value functions, does this naturally produce a rich set of options (/hierarchical RL)?

- motivate the idea as a solution to an existing problem,
!?
- prove that the "existing" problem really exists,
!?

Define HRL!?

Temoral abstractions of actions.
Actions coordinated over long time periods.
Credit assignment over long time periods.
Exploration in 'meaningful' directions.
Ok, so we wany a multiscale representation?
But also, a multiscale way to pick actions and assign credit.
And, a decompoition of ...?
Understanding how actions combine (this is necessary knowledge for HRL?)

#### Ensemble of critics. Value decomposition (in temporal scale)

Each receiving different inputs?
Or could use fourier TD to estimate. Then we can reover an FFT!?
But what else can it represent? What can vanilla TD not represent? (oscillations!?)

Relationship to something like Rudder!?

#### Multiscale state representation

If we had a multiscale state representation then we could build the policy as a fn of this representation.
Thus adding noise to the higher freq states would result in more local exploration (closer to random?!) and adding noise to the lower freq states would result in 'gobal' exploration over longer time periods!?

(huh, feels weird this has nothing to do with a heirarchical representation of the rewards)

### Scaling HRL

For tabular FuN we have;
- `manager_qs=[n_states x n_states] (current_states x goal_states)` and
- `worker_qs=[n_states x n_states x n_actions] (current_states x goal_states x actions)`.

But what if we wanted to increase the depth of the heirarchy?
- `Exectuive_manager_qs=[n_states x n_states] (current_states x exec_goal_states)` and
- `manager_qs=[n_states x n_states x n_states] (current_states x exec_goal_states x goal_states)` and
- `worker_qs=[n_states x n_states x n_actions] (current_states x goal_states x actions)`.

For every added bit of depth, the increase is upperbounded by $d \times n_{subgoals}^3$ (where $n_{subgoals} = n_{states}$) (?!?!)

__^^^__ reminds me ofsome sort of tensor factorisation!? __!!!__

But for tabular OpC. The increase is upperbounded by $n_{options}^d$. (?!?)

For tabular FuN we have;
- `qs=[n_states x n_options x actions] (current_states x options x actions)`

But what if we wanted to increase the depth of the heirarchy?
- `qs=[n_states x n_options x n_options x actions] (current_states x 1st_lvl_options x 2nd_lvl_options x actions)`


#### Making interventions at various timescales.

There exist N different scales that we can apply interventions at. We want to know;
- what these interventions do
- which interventions lead to the highest reward

What does it mean to be at a different time scale? We get access to subsampled info, or it is averaged or ...!?
Or low/high pass filters? Or !?.

#### Keys to HRL?

- Temporally abstracted actions -- (via options and Goal conditioned policies?)
- Long term credit assignment
- Heirarchical state representations
- !?

#### Equivalence

Can we prove an equivalence between options (policy conditioned value fns) and goal conditioned value fns?

$$
\begin{align}
\omega_i \in \Omega \tag{options}\\
Q(s_t, \omega_t, a) \tag{option conditioned}\\
s_g \in X \tag{goals}\\
Q(s_t, s_g, a) \tag{goal conditioned}\\
\end{align}
$$

What makes a good subgoal or termination point? How are they equivalent?

Should be able to show that they are 'dual' to each other!?

***

What are the pros/cons?
- Top of heirarchy gets reward. (Feudal)
- Bottom of heirarchy gets reward. (options) <-related to GVFs? cumulating upwards?
- All get reward?

## Possible projects

- Unsupervised options
- Equivalence of goal/option conditioned value fns
- Build a three (or even better, N) layer heirarchy
- Explore how different approaches scale (computational complexity) in the number of heirarchies
- ?

## Questions and thoughts

- Does it make sense to ask if actions can also be abstracted over dimensions other than time!?
- Relationship between learning to learn and HRL?

## Refs

- [HLMDPs](https://arxiv.org/abs/1612.02757)
- [Option-critics](https://arxiv.org/abs/1609.05140) and [Abstract options](http://papers.nips.cc/paper/8243-learning-abstract-options.pdf)
- [Modulated policy heirarchies](https://arxiv.org/abs/1812.00025)
- [Feudal](https://arxiv.org/abs/1703.01161) and [HIRO](https://arxiv.org/abs/1805.08296)(<!)
