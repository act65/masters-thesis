Ideate and pick 4 more sprints.

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
- [ ] The benefit of a heirarchy of abstractions? (versus use a single layer of abstraction). Transfer!?
- [ ] Design a new programming language. Learner gets access to assembly and must ??? (build a calculator? allow a learner to build websites easy? ...?). What would be the task / reward fn? (should be easy to learn to use, require few commands to do X, ...?)
- [ ] A single dict with the ability to merge, versus a heirarchy!?
- [ ] What is the relationship between abstraction and generalisation!?

***

> 1) What do we mean my abstraction? Let's generate some RL problems that can be exploited by a learner that abstracts.

> 2) How does abstraction actually help? Computational complexity, sample complexity, ... when doesn't it help? When it is guaranteed to help?

> 3) Can we learn a set of disentangled actions. How does that help?

> 4) How can we use an abstraction to solve a problem more efficiently? Use MPC + abstraction. Explore how different abstractions help find solutions!?


***

- Relationship between tree search and HRL? (divide and conquer for MPC) Recursive subgoal decomposition.  https://arxiv.org/pdf/1706.05825.pdf
- Absolute versus relative perspectives (relationship to subgoals and options)
