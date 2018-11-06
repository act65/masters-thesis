In IRL you observe the optimal policy and must learn the reward function. This is the inverse of the usual RL set up where, you are given the reward function and must learn the optimal policy.

***

What about just using supervision?
Observe x, a.
Predict a' = f(x). Where f(x) = argmax_a V(t(x, a)) => r(x, a, t(x, a)) + max_a' V(t(x', a')).
Train CE a.log(a').


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

# IRL:inverse generalsied policy iterationd
# the optimal policy can be
policy -> value
# invert the policy under the value fn
policy x value -> reward
```

***


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

Thus, to estimate R(s), we need to learn $V$ from $\pi$. So what do we know about $V$ that might help?

- If $\pi$ is optimal, then $V$ should be monotonoically increasing.
- ?


Not possible to recover the true reward fn!? Want to see a proof!


Count based estimation.
For every time we choose action $a$ over $a'$ not taken (is state $s$) we increment the value of that action.

$$
\begin{align}
\pi(s_t) &\sim \frac{e^{V(f(s_t, a))}}{\sum_i e^{V(f(s_t, a_i))}} \\
p(\tau) &= \prod_{t=0}^T \pi(s_t)[\hat a_t] \tag{index by actions actually taken} \\
L(\theta) &= \sum_i -\log(p(\tau_i))
\end{align}
$$

The product can be turned into a sum of log probabilities. Thus we can train per step rather than per trajectory.
