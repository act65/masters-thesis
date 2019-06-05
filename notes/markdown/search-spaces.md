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


## Indexing: Width versus Depth

You could index all possible policies with a unique id.

Given a connected MDP (defined as ...)

For all states, there must exist a sequence of actions (or policy) that (with high probability) reaches a target state from a given state.

$$
\forall s' \in S \; \exists \omega : s' = P_{\omega}(s)
$$

In a well connected MDP (defined as ?!?) the space of options is the actions $\Omega = A$.

But in a sparsely connected MDP, options are constructed from the action space.

(note: this has nothing to do with the reward so far. there may be many action sequences that yield the same option, but the reward fn will help us choose the most valueable.)


A wide MDP is a multi-armed bandit problem where each arm is an option within an MDP.


## Dynamics and connectivity

Ok, so if we parameterise our search space. We have now changed the topology of our search space (i think?!?).
If we overparameterise, then ?!? we can move between solutions in new ways!?


## Contextual decision problems

Desicions are made in stages.

But what if they were not?! Rather than adapting decision to new observations, you picked actions before you knew

- Decision problem $|A|^H$
- Contextual decision problem $|S\times A|^H$.

If adaption decisions based on history us always advantaegous, then decision problems should give an upper boun on the performance of a sequential learner.

__Q__ Is there an intermediate setting between n-armed bandits and sequential learners that controls how much information (about the past) a learner can use?


***

- Relationship to autoregressive models?
- Delay a choice for as long as possible??? Then you will have more information.
