The space we are searching makes a large difference to the efficiency of optimisation algorithms.

Smaller search spaces are better.
But adding structure (for example, locally linear) are even better.


$$
K, n \ge 2 \\
(k-1) ^ n \ge k^{n-1} \\
k^n - k^{n-1} = ??? \\
k^n - (k-1) ^ n = k \\
$$


Why is it at when solving MDPs, we do not need to optimise in the space of possible policies, but rather the space of value functions.
Why is it that

When transforming between two spaces, how does the optimisation space change?
Does my abstraction make optimisation easier?


How does knowledge about the value of one policy / action help you figure out the value ofother policies / actions?


So policy search can be reduced to ??? because
In every state, if I improve my action for more reward / value then it increases the value of the policy globally.
We can decompose it into many local value based problems, rather than optimising over possible policies.


$$
\mathop{\text{max}}_V \mathop{\mathbb E}_{s\sim D} V(s) \\
\mathop{\text{max}}_{\pi} \mathop{\mathbb E}_{s\sim D}V^{\pi}(s) \\
\mathop{\text{max}}_{\theta} \mathop{\mathbb E}_{s\sim D} V_{\theta}(s) \\
\mathop{\text{max}}_{\theta} \mathop{\mathbb E}_{s\sim D} V^{\pi_{_{\theta}}}(s) \\
\mathop{\text{max}}_{\phi} \mathop{\mathbb E}_{s\sim D} V^{\pi_{_{\theta_{\phi}}}}(s) \\
\mathop{\text{max}}_{\varphi} \mathop{\mathbb E}_{s\sim D} V^{\pi_{_{\theta_{\phi_{\varphi}}}}}(s) \\
$$

We can pick the space we optimise in. Why would we want to pick one space over another?

- In which spaces does momentum work well?
- In which spaces can we do convex optimisation?
- In which spaces ...

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
  policy = greedy_update(value)
```

### Parameter iteration

```python
parameters = init()
while not converged(value):
  policy = fn(parameters)
  value = evaluate(policy)
  parameters = greedy_step(value)
```
