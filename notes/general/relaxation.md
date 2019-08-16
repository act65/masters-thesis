https://en.wikipedia.org/wiki/Linear_programming_relaxation
https://en.wikipedia.org/wiki/Randomized_rounding

> The basic approach has three steps:
- Formulate the problem to be solved as an integer linear program (ILP).
- Compute an optimal fractional solution $x$ to the linear programming relaxation (LP) of the ILP.
- Round the fractional solution $x$ of the LP to an integer solution $x'$ of the ILP.


***

A relaxation of the minimization problem

$$
z=\min\{c(x):x\in X\subseteq R^n
$$

is another minimization problem of the form

$$
z=\min\{c_R(x):x\in X_R \subseteq R^n
$$

with these two properties

1. $X_R \supseteq X$
2. $c_R(x)\leq c(x) \forall x\in X$

The first property states that the original problem's feasible domain is a subset of the relaxed problem's feasible domain. The second property states that the original problem's objective-function is greater than or equal to the relaxed problem's objective-function.
https://en.wikipedia.org/wiki/Relaxation_(approximation)

***

1. Let $\Pi: S \to A$ be the space of policies. And $\mathcal P: S \times A \to \Delta S$ be the space of action conditioned transition fns. And $\mathcal U: S \to \Delta S$ be the space of state transition fns. Then $\mathcal P \times \Pi \supseteq \mathcal U$.
2. ???

So if we change everything for the maximisation case. And (2) is satisfied, then the value returned by the LMDP is a lower bound on the expected value of the MDP. 

$$
V_{MDP}(x) \ge V_{LMDP}(x)
$$
