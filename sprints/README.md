I have allocated time for 8 'sprints', each of 2 weeks. The goal of each sprint will be to;

- motivate the idea as a solution to an existing problem,
- prove that the "existing" problem really exists,
- generate alternative solutions and a suitable baseline,
- design the minimal viable experiment to falisify the proposed solution,
- implement the experiment if feasible.

### The sprints

- Differentiable decompositions.
- Temporal meta-RL and lr2rl.
- Reachability.
- Density based transfer.
- How can we train modular systems when there is often no gradient defined (as the module was not used)? Using 'counterfactual' credit assignment of what might have happened?
- Build a [differentiable neural computer]() with locally structured memory (start with 1d and then generalise to higher dimensions). Is the ability to localise oneself necessary to efficiently solve partial information decision problems? Under which condtions does the learned index to a locally structured memory approximate the position of the agent in its environment.
- When attempting to learn a model, the agent uses an exploration policy. This policy may influence the dynamics observed, thus we need to use off-policy methods to correct for the effects of exploration actions. (The model must somehow disentangle the agents policy, and its effects, from the dynamics of the system)
- Inverse energy learning. Inspired to [inverse reinforcement learning](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf), what if we assume that the observations we make are the results of some optimal action, in the case of an energy being minimised, $\Delta x = -\eta\frac{\partial E}{\partial x}$.
