There are a fw different parts to the complexity!?

- Discovery of the abstraction
- Solving the abstraction
- Mapping back to the original

Each can be split into sample and computational complexity. (??) Although they are related... one depends on the other.

## Qs and Ts

- Comparing complex seems superhard!? If one is cts and the other discrete then we can use different algorithms. So, Q: in general which representations allow the use of different optimisation algols? It doesnt seem sufficient to solve it for a single algol... But first step blah blah...
- Dont just care abou finding the optimal solution, but also $\epsilon$ accurate solutions. How has the space of these changed? Is it now disconnected? Or does it have less volume? Etc...
- (when) Is finding the optimal policy harder in the abstraction than in the original space, even if value is preserved!??


### Computational complexity
(problem is that for computational complexity, the details of the algol become important.
  when analysing approximation bounds we dont have to consider how to find the approximations...)

MPD + GPI.

- Need to show how quickly the evaluations converge (possibly under a non-stationary policy).
- Need to show that updating $\pi$ with an $\epsilon$ accurate V yeilds some expected step size.
- Need to show these converge.

MDP + PG.



## Computational complexity

#### Resolving goals

Just imagine the lazy heirarchical MPC learner with arbitrary goals / rewards.
The 'hierarchical' approach still provides benefits (??).
It solves every possible state-subgoal pair and knows how to generate paths between all states. This knowledge will allow faster transfer to any task in the same environment.

No, not sure this works.

But what does the heirarchy provide here?!?
This canbe done with a single layer...
