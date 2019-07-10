Goals.

- Understand MDPs and their properties
- Which properties are important?
- How can we put metrics on these properties? And analyse in higher dims?
- ?

Todos.

- Generalise the types of parameterised fns (jax must have some tools for this)
- calculate the tangent fields. are they similar? what about for parameterised versions, how can we calculate their vector fields???

Questions
- Is param + momentum dynamics are more unstable? Or that you move around value-space in non-linear ways??
- Is param + momentum only faster bc it is allowed larger changes? (normalise for the number of updates being made). Answer. No. PPG is still faster.
- What if we make the learning rate very small? (!!?!) (plot momentum as a fn of different lrs. w same init / mdp)
- What is the max difference between a trajectory derived from cts flow versus a trajectory of discretised steps on the same gradient field?

Observations

- Parameterisation seems to accelerate PG, but not VI. Why?
- Sometimes VI converges to a weird place.
-
