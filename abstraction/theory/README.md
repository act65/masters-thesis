How does HRL actually help?
Why should we care?

Want to prove that HRL can give exp speed up of learning complexity over RL.

***

Want to show that;

- accurate: action abstraction can recover (within $\epsilon$) the optimal solution.
- efficient: the complexity of finding the optimal solution (compared to the original) is lower?

But these seem totally banal!?

Want a tool that takes a representation (of the action space - or state space?) and dreives/estimates bounds on its accuracy and complexity.


## Possible settings/approaches

 - Original problem -> abstraction -> abstract optimal policy -> difference between abstract and original optimal policies
 - Explore the loss surface of a linear RL problem and explore how it gets transformed under an abstraction
 - Frame as finding a set of spanning edge types for a graph. (?)


## References

- [Towards a Unified Theory of State Abstraction for MDPs](http://anytime.cs.umass.edu/aimath06/proceedings/P21.pdf)
- [Near Optimal Behavior via Approximate State Abstraction](https://arxiv.org/abs/1701.04113)
- [State Abstractions for Lifelong Reinforcement Learning](http://proceedings.mlr.press/v80/abel18a.html)
