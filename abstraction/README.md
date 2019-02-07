Ideate and pick 4 more sprints.

> 1) What do we mean my HRL? Let's generate some RL problems that can be exploited by a hierarchical learner.


> 2) How does HRL actually help? Computational complexity, sample complexity, ... when doesn't it help? When it is guaranteed to help?

> 3) Can we learn a set of disentangled actions. How does that help?

> 4)




- [ ] Unsupervised options (how good can random ones be!?)
- [ ] Equivalence of goal/option conditioned value fns
- [ ] Build a three (or even better, N) layer heirarchy
- [ ] Explore how different approaches scale (computational complexity) in the number of layers in the heirarchy
- [ ] Use a learned reachability metric to measure proximity to subgoals (and thus use to give rewards)
- [ ] A heirarchical subgoal net that uses MPC rather than learned policies
- [ ] Explore function approximation for options a = f(w) (rather than look up table)
- [ ] How does this related to a decomposition of the value function?
- [ ] How to achieve stable training of a hierarchy?
- [ ] Filtering / gating state space to the lower levels
- [ ] Connection to evolution of language.
