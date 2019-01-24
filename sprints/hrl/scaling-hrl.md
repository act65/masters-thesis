#### Scaling HRL

For __tabular FuN__ we have;
- `manager_qs=[n_states x n_states] the (current_states x goal_states)` and
- `worker_qs=[n_states x n_states x n_actions] the (current_states x goal_states x actions)`.

But what if we wanted to increase the depth of the heirarchy?
- `Exectuive_manager_qs=[n_states x n_states] the (current_states x exec_goal_states)` and
- `manager_qs=[n_states x n_states x n_states] the (current_states x exec_goal_states x goal_states)` and
- `worker_qs=[n_states x n_states x n_actions] the (current_states x goal_states x actions)`.

For every added layer of depth, the increase complexity is upperbounded by (d = depth) $d \times n_{subgoals}^3$ (where $n_{subgoals} = n_{states}$) (?!?!)

(__^^^__ reminds me of some sort of tensor factorisation!? __!!!__)

But for __tabular OpC__. For tabular FuN we have;
- `qs=[n_states x n_options x actions] the (current_states x options x actions)`

But what if we wanted to increase the depth of the heirarchy?
- `qs=[n_states x n_options x n_options x actions] the (current_states x 1st_lvl_options x 2nd_lvl_options x actions)`
- and we would need to compute integrals of each layer of depth (so memory and time complexity are bad)

The increase is upperbounded by $n_{options}^{d}$. (?!?)

__BUT.__ What about scaling with state size?

- OpC - $O(n)$
- FuN - $O(n^3)$
