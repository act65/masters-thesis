Todos.

- Verify these alogls are converging to the optimal policy. (plot the polytope)
- Generalise the parameterised fns (jax must have some tools for this)
- Implement policy gradients.


What if we;
- normalise for the number of updates being made. (Is parameterised only faster bc it is allowed larger changes?)
- make the learning rate very small? (!!?!)


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
