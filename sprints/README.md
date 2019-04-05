I have allocated time for 8 'sprints', each of 2 weeks. The goal of each sprint will be to;

- motivate the idea as a solution to an existing problem,
- prove that the "existing" problem really exists,
- generate alternative solutions and a suitable baseline,
- design the minimal viable experiment to falisify the proposed solution,

### The sprints

- [x] Differentiable decompositions: metrics for disentanglement.
- [ ] Differentiable decompositions: modular credit assignment.
- [ ] Causal RL. OR Temporal credit assignment. OR MBRL...
- [ ] Memory (locally structured)
- [ ] Exploration (effects).
- [ ] Reachability.
- [x] Inverse energy learning.
- [x] Heirarchical RL.


***

> 5.Build a differentiable neural computer (Graves et al. 2016) with locally structured memory (start with 1d and then generalise to higher dimensions). Is the ability to localise oneself necessary to efficiently solve partial information decision problems? Under which conditions does the learned index to a locally structured memory approximate the position of the agent in its environment.


Memory structures + Locality.

- https://www.nature.com/articles/nn.4661.pdf
- https://arxiv.org/pdf/1602.03218.pdf
- https://arxiv.org/pdf/1609.01704.pdf
- DNC


***

An alternative view on disentanglement. But what is its relationship to independence?

- Counterfactual estimates of gradients for modules?
- Blackbox jacobian sensing


We want n different modules that specialise in their tasks? How can we learn different specialists?

- winner takes all credit assignment. positive feedback/lateral inhibition/...?
- give access to different inputs/resources. they physically/computationally cannot do the same thing...
- train on different tasks...
- ?


***

Ok, imagine we give two specialists different inputs (say, left-right halves of mnist) and train them separately on the same classification task. Their outputs are __not__ going to be independent... The fact the the left half classifies its input as a 2, makes it likely that the right half will also classify its input as a 2.

So, in this instance, the two modules will output independent results if;

- they are trained on the same task yet receive independent inputs
- the two experts are trained on different tasks, which are independent
- ?

***

Questions
- __Specialisation__! Independently useful contributions.
- __Q__ How is independence related to decomposition?

## Resources

- [MoE](https://arxiv.org/abs/1701.06538) <-- could measure the hidden states MI/TC!?
- [DDP for Structured Prediction](https://arxiv.org/pdf/1802.03676.pdf)
