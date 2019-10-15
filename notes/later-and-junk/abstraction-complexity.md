There are a few different parts to characterising the 'complexity' of an abstraction.

- Discovery of the abstraction
- Solving the abstraction
- Mapping back to the original

What do we mean by complexity?

Sample,
computational complexity (memory, time, ?)

Also, what do we mean by worst case? Worst case with probability < X?


## Qs and Ts

#### General computational complexity

- Comparing complexity of algols seems superhard!? If one is cts and the other discrete then we can use different algorithms. So, Q: in general which representations allow the use of different optimisation algols? It doesnt seem sufficient to solve it for a single algol... But first step blah blah... Instead use lower bounds?!
- ?

#### Abstraction computational complexity

- Dont just care abou finding the optimal solution, but also $\epsilon$ accurate solutions. How has the space of these changed? Is it now disconnected? Or does it have less volume? Etc...
- (when) Is finding the optimal policy harder in the abstraction than in the original space, even if value is preserved!??


### Upper bound

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


***

Want to charachterise the convergence of $V_A^{\pi_t} \to V_G^{\pi^* }$ under $\pi_t = \pi_{t-1} - \nabla J()$.
Want show that the error decreases as a funtion of time steps. $\mathcal O(\frac{1}{t^2})$
