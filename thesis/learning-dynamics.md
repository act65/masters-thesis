

## Learning (dynamics / Complexity)

- Partitioning policy space.
- jumping from corner to corner
- two $\epsilon$ different policies. why do they have distinctly different dynamics?!
- number of steps (in high n_actions?)


__Q:__ (how) Does it help?

Complexity of finding the abstraction + solving vs solving ground problem.


Want to show that. The convergence rate / amount of data required is less for the abstraction.

$$
\begin{align}
e_{G} &= \mathcal O(n, |A|, t) \\
e_{A} &= \mathcal O(m, |A|, t) \\
\therefore & e_{G} \ge e_{A}
\end{align}
$$


## Incremental updates

Given $P_0, R_0$.
At each time step we recieve $\Delta R, \Delta P$.
Want to efficiently calculate $\Delta V$ s.t. $V_t = \Delta V + V_{t-1}$.
Oh... Gradient descent.

So how much computation does this incremental update save us?
Are there more efficient ways?

But want to use the information as soon as possible. Offline training means that the data can be 'stale'. Can this be quantified.
Related to distributed optimisation!?

$$
dV = (I-\gamma dP \pi)^{-1}dr \pi
$$



***

- What can go wrong with abstracting? Imagine a simple linear cts relationship between state and value. We then discretise this and must learn the value independently for each of the discretised points. Can be much worse...
- General problem. Need the transition fn / reward fn to solve analytically. Aka we are doing model based RL. BUT. It has a few downfalls. How does model free and DRL get around these issues!? Model-free is appealing because, ?!?.


![The gradient of the value with repspect to the policy for random MDPs](../pictures/polytope-vector-fields.png)
