Discounts and uncertainty about the future (not considering temporal credit assignment)

### Machine behaviour

Lets just build some intutiion for what the discount does, what it controls, etc.

Define depression as:
Define succeptibility to addiction (/ impulsiveness) as:


Want some intuitive examples

## Discounting

__Q:__ Why do we discount? Could use the average reward.

__TODO__ Want to derive the need for a discount when there is noise within the model used by a planner.


### Discounting for uncertainty

$$
\begin{align}
Q(s_t, a_t) &= r(s_t, a_t) + \gamma \mathop{\mathbb E}^{s' \sim P(\cdot | s_t, a_t)}_{a\sim \pi(\cdot| s')} Q(s',a)\\
Q(s_t, a_t) &= r(s_t, a_t) + \gamma \sum_{s'}P(s'|s_t, a_t)\sum_a \pi(a | s') Q(s',a)\\
\end{align}
$$

##### Policy entropy

$$
\begin{align}
Q(s_t, a_t) &= r(s_t, a_t) +  \sum_{s'}P(s'|s_t, a_t) \gamma(s_t)\sum_a \pi(a | s') Q(s',a)\\
\gamma(s) &= 1-\mathop{\mathbb E}_{a\sim\pi(\cdot | s)}[-\log(\pi(a | s))]\\
\end{align}
$$

For many possible actions, this could be expensive to calculate at every step.

Can this still be solved as a set of linear equations?

> If the [policy] is low entropy, then you should be able to significantly increase the discount factor. This is because the noise of policy gradient algorithms scales with the amount of information that is injected into the trajectories by sampling from the policy. [Alex Nichol](https://blog.aqnichol.com/2019/04/03/prierarchy-implicit-hierarchies/)

Take a minimum entropy policy, $H(\pi) = 0$, then $\gamma = 1$. This is a deterministic policy, yielding zero ... making it far easier to plan.

##### Transition entropy

$$
\begin{align}
Q(s_t, a_t) &= r(s_t, a_t) +  \gamma(s_t, a_t)\sum_{s'}P(s'|s_t, a_t) \sum_a \pi(a | s') Q(s',a)\\
\gamma(s, a) &= 1-\mathop{\mathbb E}_{a\sim P(\cdot | s, a)}[-\log(P(s' | s, a))]\\
\end{align}
$$


## Discount rates

Why exponential discounting?
