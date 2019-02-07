How does HRL actually help?
Why should we care?

Want to prove that HRL can give exp speed up of learning complexity over RL.

- Data complexity speed up?
- Computational complexity speed up?

Existing work on abstraction?

In the case that the abstraction is isomorphic to ??? then no problem.
Want to measure the optimal policies return w and w/o abstraction. The quality of the abstraction is then then difference in computational/sample complexity divided by the change in optimal return.
(ahh, no. bc the optimal policy might now be a lot harder to find!?!)

## Questions

- (when) Is finding the optimal policy harder in the abstraction than in the original space, even if value is preserved!??
-

## Computational complexity

#### Resolving goals

Just imagine the lazy heirarchical MPC learner with arbitrary goals / rewards.
The 'hierarchical' approach still provides benefits (??).
It solves every possible state-subgoal pair and knows how to generate paths between all states. This knowledge will allow faster transfer to any task in the same environment.

No, not sure this works.

But what does the heirarchy provide here?!?
This canbe done with a single layer...

#### ???


## Possible settings/approaches

 - Original problem -> abstraction -> abstract optimal policy -> difference between abstract and original optimal policies
 - Explore the loss surface of a linear RL problem and explore how it gets transformed under an abstraction
 - Frame as finding a set of spanning edge types for a graph. (?)

## References

- [Towards a Unified Theory of State Abstraction for MDPs](http://anytime.cs.umass.edu/aimath06/proceedings/P21.pdf)
- [Near Optimal Behavior via Approximate State Abstraction](https://arxiv.org/abs/1701.04113)
- [State Abstractions for Lifelong Reinforcement Learning](http://proceedings.mlr.press/v80/abel18a.html)
