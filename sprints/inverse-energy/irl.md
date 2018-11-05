In IRL you observe the optimal policy and must learn the reward function. This is the inverse of the usual RL set up where, you are given the reward function and must learn the optimal policy.

```python
reward -> policy (x value)
policy -> reward (x value)
```

### IRL

> Given an observed policy, we can generate a point estimate of the reward function from the posterior distribution over reward functions. To construct this point estimate, we must know the likelihood of observing each policy for each given reward function

Two sources of uncertainty? -- no that's the same source...?
A single reward function can generate many optimal policies (when?).
An optimal policy can be optimal in many ways.


What about just using supervision?
Observe x, a.
Predict a' = f(x). Where f(x) = argmax_a V(t(x, a)) => r(x, a, t(x, a)) + max_a' V(t(x', a')).
Train CE a.log(a').

What is the opposite of diffusion? Is it possible to run dynamic programming backwards in time?

$$
\begin{align}
V(s) &= R(s) + \gamma E_{s'\sim p(s, \pi^* (s))} [ V(s') ] \\
R(s) &= V(s) - \gamma E_{s'\sim p(s, \pi^* (s))} [ V(s') ] \\
&\approx \frac{V(s) - V(s')}{ds} ??\\
\end{align}
$$

Is it possible to learn $V$ and $R$ from $\pi$?

If $\pi$ is optimal, then $V$ should be monotonoically increasing.


Not possible to recover the true reward fn!?
