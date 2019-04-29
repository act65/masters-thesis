Experiments

- [x] `reproduce_polytope.py`: Simply reproduce the figures from [The Value Function Polytope in Reinforcement Learning](https://arxiv.org/abs/1901.11524)
- [ ] `polytope_entropy.py`: How are policies distributed in value space?
  - [x] Visualise the density of the value function polytope.
  - [ ] Calculate the expected suboptimality (for all policies - and all possible Ps/rs)? How does this change in high dimensions?
  - [ ] Apply an abstraction (with X property) and visualise how the distribution of value changes.
- [ ] `gpi_partitions.py`: How does generalised policy iteration partition the policy space?
  - [x] Visualise how the number of steps required for GPI partitions the policy / value spaces.
  - [ ] Generalise GPI to work in higher dimensons. Calclate how does it scales.

***

- [ ] `reproduce_abstraction.py`: Simply reproduce the figures from [Near Optimal Behavior via Approximate State Abstraction](https://arxiv.org/abs/1701.04113)
- [ ] `symmetry_detection.py`: Reproduce results from [Theja](???)
- [ ] `disentangled_actions.py`: ??/

***

Other possible experiments

- [ ] `gpi_topology.py` Generate a graph of update transitions under GPI (or another update fn). The nodes will be policies the edges wil be greedy steps using the estimated value.
- [ ] `discount_basis_state_values.py`
- [ ] Action value, V(a). When action can be applied at all states, we can take the dual of the MDP, where the actions and state spaces are swapped. How does this effect the value? Can a solution to one be transferred to the other.
- [ ] Verify that the distribution over values (under uniform policies) is also uniform. (random matrix theory flavour)
- [ ] `action_meta_learning.py` Given a very large space of actions (or actions that are changing). Want to be able to quickly figure out what the action does.
- [ ] `symmetric_action.py` Construct a measure of similarity based on ???
