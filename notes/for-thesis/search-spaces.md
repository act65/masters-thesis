## Search spaces

When optimising MDPs, would we rather optimise within the set of potentially optimal policies, the $|A|^{|S|}$ discrete policies, or would we rather optimise within the set of possible value functions, $\mathbb R^{|S|}$, or ...?

__Q:__ How does the space we are searching within effect the search dynamics? Convergence, trajectories, ...

Smaller search spaces are better. But added structure (for example, locally linear) can be exploited.

When transforming between two spaces, how does the optimisation space change?
Does my abstraction make optimisation easier?

$$
\begin{align}
&\mathop{\text{max}}_V \mathop{\mathbb E}_{s\sim D} V(s) \\
&\mathop{\text{max}}_{\pi} \mathop{\mathbb E}_{s\sim D}V^{\pi}(s) \\
&\mathop{\text{max}}_{\theta} \mathop{\mathbb E}_{s\sim D} V_{\theta}(s) \\
&\mathop{\text{max}}_{\theta} \mathop{\mathbb E}_{s\sim D} V^{\pi_{_{\theta}}}(s) \\
&\mathop{\text{max}}_{\phi} \mathop{\mathbb E}_{s\sim D} V^{\pi_{_{\theta_{\phi}}}}(s) \\
&\mathop{\text{max}}_{\varphi} \mathop{\mathbb E}_{s\sim D} V^{\pi_{_{\theta_{\phi_{\varphi}}}}}(s) \\
\end{align}
$$

We can pick the space we optimise in. Why would we want to pick one space over another?

- In which spaces can we go gradient descent?
- In which spaces can we do convex optimisation?
- In which spaces does momentum work well?
- ...

### Value iteration

```python
value = init()
while not converged(value):
  value = update(value)
```

### Policy iteration

```python
policy = init()
while not converged(value):
  value = evaluate(policy)
  policy = update(value)
```

### Parameter iteration

```python
parameters = init()
while not converged(value):
  policy = fn(parameters)
  value = evaluate(policy)
  parameters = step(value)
```

## Dynamics and connectivity

Ok, so if we parameterise our search space. We have now changed the topology of our search space. See these gradient flows for example;

![]()

If we overparameterise the search space, then we can move between solutions in new ways. We can 'tunnel' from A to B, without crossing C.

What advantages does this provide?

Every point is closer, under some measure of distance?!?


But. Momentum seems like it might be a bad thing here?

## Momentum in higher dimensions

Intuition. Something weird happens with momentum in overparameterised spaces.

It is necessary to consider the trajectory to study momentum. It depends on what has happened in the past.
Can we construct a space of possible trajectories?
What properties do trajectories have? They are connected by the update fn.

## Continuous flow and its discretisation

A linear step of size, $\alpha$, in parameter space, ie by gradient descent, is not necessrily a linear step in parameter space.
