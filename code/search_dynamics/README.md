Todos.

- Verify these alogls are converging to the optimal policy. (plot the polytope)
- Generalise the parameterised fns (jax must have some tools for this)
- Implement policy gradients.

Questions
- Is param + momentum dynamics are more unstable? Or that you move around value-space in non-linear ways??
- Is param + momentum only faster bc it is allowed larger changes? (normalise for the number of updates being made)
- What if we make the learning rate very small? (!!?!)
- How can we put metrics on these? And analyse in higher dims? Which properties do we care about?

***

```python
solvers = [
    value_iteration,
    policy_iteration,
    parameter_iteration
]

optimisers = [
    sgd,
    momentum
]
```
