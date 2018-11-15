## IRL

In IRL you observe the optimal policy and must learn the reward function. This is the inverse of the usual RL set up where, you are given the reward function and must learn the optimal policy.

> The entire field of reinforcement learning is founded on the presupposition that the reward function, rather than the policy, is the most succinct, robust, and transferable definition of the task. [ref](https://github.com/sjchoi86/irl_rocks/blob/master/IRL_survey.pdf)

__Q:__ ^^^ Is this really true!? In what cases?

## Generalised policy iteration

Almost all RL proceeds using GPI. Summarised below.

```python
### RL: generalised policy iteration
# evaluate the current policy under the given reward
reward x policy -> value
# update the current policy
value -> policy
```


### Policy values (rationalise actions)

> Goal: estimate $V$ from $\pi^{* }$.

What do we know about $V$ that might help?

1. The actions not taken should have lower $V$ than the actions taken.
2. If $\pi$ is optimal, then $V$ should be monotonoically increasing over time.


$$
\begin{align}
\pi(s_t) &\sim \frac{e^{V(f(s_t, a))}}{\sum_i e^{V(f(s_t, a_i))}} \\
p(\tau) &= \prod_{t=0}^T \pi(s_t)[\hat a_t] \tag{index by actions actually taken} \\
L(\theta) &= \sum_i -\log(p(\tau_i))
\end{align}
$$

The product can be turned into a sum of log probabilities. Thus we can train per step rather than per trajectory. $p(\tau)$ can be decomposed as we have access to the true state $s_t$.


### Inversion

> If we have the value fn and the optimal policy then how can we estimate the reward?

What is the opposite of diffusion? Is it possible to run dynamic programming backwards in time?

$$
\begin{align}
V(s) &= R(s) + \gamma E_{s'\sim p(s, \pi^* (s))} [ V(s') ] \tag{policy evaluation}\\
R(s) &= V(s) - \gamma E_{s'\sim p(s, \pi^* (s))} [ V(s') ] \tag{inverse policy evaluation}\\
&= V(s) - \gamma V(s') \tag{deterministic policy/transition} \\
&= V(s) - V(s + \Delta s) \tag{$\gamma \approx 1$, rewrite $s$}\\
&= \frac{d V}{d s}\Delta s \tag{!?!?} \\
\\
V(s_0) &= \sum^{\infty}_{t=0} \gamma^t R(s_t) \\
&= \int_0^{\infty} \gamma^t R(s(t)) dt \\
&= \int_S \gamma^t R(s) ds \\
\frac{dV(s)}{ds} &=  \gamma^t R(s) \tag{!!?!}\\
\end{align}
$$
(where $\Delta s$ is a specific change in $s$ along the direction of the best action)

### Generalised value iteration.

We could start with a simple guess (eg initially a constant?). Would need to show that the value converges, and so does the reward.

```python
# IRL: generalsied inverse policy iteration
# invert the policy under the current value fn
policy x value -> reward
# evaluate
policy x reward -> value
```

### A metric

- Optimal policy is invariant to affine transformation (and scalings?) of the reward (and value)
- Two reward fns can be the same except for a single state, and produce vastly different optimal policies.

So rather than measuring the difference between the learned reward (which assumes we have access to it...). We could evaluate by comparing the optimal policies under the two rewards.

But how can two policies be compared?

Problems;
- that there might be two optimal policies. And one could give high similarity to the learned policy, and the other might not.
- a single action can be very important.

Ok, so we should compare the value fns?

> Theorem: Two optimal policies might be different but must have the same value function. [need to find ref]

Sure. Makes sense. Except that we dont have the true value function...



### Matching feature expectation

Ref - [Apprenticeship Learning via IRL](https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf)

$$
\begin{align}
R(s) &= w^T \phi(s)\\
V^{\pi}(s) &= w^T\mathbb E [\sum \gamma^i\phi(s_i)] \\
&= w^T \mathbb E [\Big(\phi(s_i) + \gamma \mu^{\pi}(\tau(s_i,a_i))\Big)] \\
R(s,a) &= w^T \phi(s, a)\\
Q^{\pi}(s, a) &= w^T \mathbb E [\sum \gamma^i\phi(s_i, a_i)] \\
&= w^T \mathbb E [\Big(\phi(s_i, a_i) + \gamma \mu^{\pi}(s_{i+1},a_{i+1})\Big)] \\
\end{align}
$$

Either:
- Need access to the transition function.
- The future state-action pair.

This is quite nice! Now we need to learn $w$ and $\phi$ and $\mu$ (note we could learn $\phi$ in an unsupervised fashion). Only need to learn $V$ and we get R for free.

## Thoughts

- Not possible to recover the true reward fn!? Want to see a proof!
- __Q:__ Oh. Is it not possible for IRL to just observe!? Does it need to act as well? Test out rewards and see what policies are generated, and then compare them to the optimal observations?
- Hmph. What about getting access to non-optimal trajectories. It seems that this could increase the speed of learning (if the non-optimal trajectories were chosen correctly). We receive pairs indicating the dicision boundaries, good-bad. We dont get access to any negative samples. How does this restrict out ability to learn and generalise!?

## Refs

- [IRL](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf)
- [Apprenticeship via IRL](https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf)
- [Max entropy IRL](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf)
- [DAC for IRL](https://arxiv.org/abs/1809.02925)
- [IOC](https://arxiv.org/abs/1805.08395)
- [Review](https://arxiv.org/abs/1806.06877)
- [IOC with LMDPs](https://homes.cs.washington.edu/~todorov/papers/DvijothamICML10.pdf)
- [GIRL](https://pdfs.semanticscholar.org/6021/4094bb268d137f021fdff10c298fc92cde33.pdf)
- https://github.com/sjchoi86/irl_rocks/blob/master/IRL_survey.pdf !!!
