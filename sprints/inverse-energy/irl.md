In IRL you observe the optimal policy and must learn the reward function. This is the inverse of the usual RL set up where, you are given the reward function and must learn the optimal policy.

### IRL

> Given an observed policy, we can generate a point estimate of the reward function from the posterior distribution over reward functions. To construct this point estimate, we must know the likelihood of observing each policy for each given reward function

Two sources of uncertainty? -- no that's the same source...?
A single reward function can generate many optimal policies (when?).
An optimal policy can be optimal in many ways.

***


```python
### RL: generalised policy iteration
# evaluate the current policy under the given reward
reward x policy -> value
# update the current policy
value -> policy

# IRL: generalsied inverse policy iteration
# the optimal policy can be
policy -> value
# invert the policy under the current value fn
policy x value -> reward
```

### Policy values

> Goal: estimate $V$ from $\pi^{* }$.

What do we know about $V$ that might help?

- The actions not taken should have lower $V$ than the actions taken.
- If $\pi$ is optimal, then $V$ should be monotonoically increasing over time.


$$
\begin{align}
\pi(s_t) &\sim \frac{e^{V(f(s_t, a))}}{\sum_i e^{V(f(s_t, a_i))}} \\
p(\tau) &= \prod_{t=0}^T \pi(s_t)[\hat a_t] \tag{index by actions actually taken} \\
L(\theta) &= \sum_i -\log(p(\tau_i))
\end{align}
$$

The product can be turned into a sum of log probabilities. Thus we can train per step rather than per trajectory. $p(\tau)$ can be decomposed as we have access to the true state $s_t$. We can use the markov property to ensure that ...


***

Simplest algol. Use discounted counts. Matching feature expectation.

$$
V^{\pi}(s) = w^T\sum \gamma^i\phi(s_i)
$$
https://github.com/sjchoi86/irl_rocks/blob/master/IRL_survey.pdf !!!

### Inversion

What is the opposite of diffusion? Is it possible to run dynamic programming backwards in time?

$$
\begin{align}
V(s) &= R(s) + \gamma E_{s'\sim p(s, \pi^* (s))} [ V(s') ] \tag{policy evaluation}\\
R(s) &= V(s) - \gamma E_{s'\sim p(s, \pi^* (s))} [ V(s') ] \tag{inverse policy evaluation}\\
&= V(s) - \gamma V(s') \tag{deterministic policy/transition} \\
&= V(s) - V(s + \Delta s) \tag{$\gamma \approx 1$, rewrite $s$}\\
&= \frac{d V}{d s} \tag{!?!?} \\
\\
V(s_0) &= \sum^{\infty}_{t=0} \gamma^t R(s_t) \\
&= \int_0^{\infty} \gamma^t R(s(t)) dt \\
\frac{ds(t)}{dt} &= f(s(t), a^* ) - s(t) \\
\end{align}
$$
(where $\Delta s$ is a specific change in $s$ along the direction of the best action)

***

- __Generalised reward iteration.__ What about `policy x reward' -> value`, `policy x value -> reward`. Where `reward'` is a simple guess (eg initially a constant?). Would need to show that the value converges, and so does the reward.
- __End-to-end.__ Could use GPI and make sure the whole thing is end-to-end differentiable.


### A metric

- Optimal policy is invariant to affine transformation (and scalings?)
- Two reward fns can be the same except for a single state, and produce vastly different optimal policies.

So rather than measuring the difference between the learned reward (which assumes we have access to it...). We could evaluate by comparing the optimal policies under the two rewards.

But how can two policies be compared?

Problems;
- that there might be two optimal policies. And one could give high similarity to the learned policy, and the other might not.
- a single action can be very important.

Ok, so we should compare the value fns?

> Theorem: Two optimal policies might be different but must have the same value function. [need to find ref]

Sure. Makes sense. Except that we dont have the true value function...


***

> The entire field of reinforcement learning is founded on the presupposition that the reward function, rather than the policy, is the mostsuccinct, robust, and transferable definition of the task. [ref](https://github.com/sjchoi86/irl_rocks/blob/master/IRL_survey.pdf)

Is this really true!? In what cases?

***

Not possible to recover the true reward fn!? Want to see a proof!


## Refs

- [IRL](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf)
- [Apprenticeship via IRL](https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf)
- [Max entropy IRL](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf)
- [DAC for IRL](https://arxiv.org/abs/1809.02925)
- [IOC](https://arxiv.org/abs/1805.08395)
- [Review](https://arxiv.org/abs/1806.06877)
- [IOC with LMDPs](https://homes.cs.washington.edu/~todorov/papers/DvijothamICML10.pdf)
- [GIRL](https://pdfs.semanticscholar.org/6021/4094bb268d137f021fdff10c298fc92cde33.pdf)
