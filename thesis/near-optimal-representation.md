We are working with MDPs $(S, A, \tau, r)$, therefore we have a state space, $S$, an action space, $A$, a transition function $P: S\times A \times S \to [0, 1]$ and a reward function $r: S\times A \to \mathbb R$.

## Near optimal abstractions

I am interested in reasoning about which policies are representable within an abstracted MDP. We care about showing that an abstraction with certain properties can approximately represent the optimal solution to the original problem.

An abstract MDP is defined as;

<!-- (Must be smaller / lower complexity than the original) -->

The metric we are optimising is the representation error of the optimalpolicy. Given an abstraction, we want to knowhow well the abstraction can represent the optimal policy.

$$
\forall_{s\in S_G, a\in A_G} \mid Q_G^{\pi^* }(s, a) - Q_G^{\pi_{GA}^* }(s, a) \mid \le 2 \epsilon \eta_f
$$

We could impose properties on a state abstraction using something like the following;

$$
\begin{align}
\forall_{s_1, s_2 \in S} \mid f(s_1) - f(s_2)\mid \le \epsilon &\implies \phi (s_1) = \phi(s_2)\\
\end{align}
$$

In other words, if there exists an approximate similarity, according to $f$, then build it into our abstraction.

- __Q:__ How should we construct our abstraction?
- __Q:__ What properties should it have to achieve 'good' performance?

Using the above method of imposing properties on an abstraction, what should we pick as $f$?

1. The policy function: $\forall_{\cdot_a, \cdot_b \in D} \mid \pi(\cdot_a) - \pi(\cdot_b) \mid \le \epsilon$ is approximately the same.
1. The transition function: $\forall_{\cdot_a, \cdot_b \in D} \mid \tau(\cdot_a) - \tau(\cdot_b)\mid \le \epsilon$ is approximately the same.
1. The reward function: $\forall_{\cdot_a, \cdot_b \in D} \mid r(\cdot_a) - r(\cdot_b) \mid \le \epsilon$ is approximately the same.


Also,

4. The policy trajectory: $\forall_{\cdot_a, \cdot_b \in D} \mid \sum_{t=0}^T \parallel \pi(\cdot_a) - \pi(\cdot_b)\parallel_1 \mid \le \epsilon$ is approximately the same.
1. The transition trajectory: $\forall_{\cdot_a, \cdot_b \in D} \mid \sum_{t=0}^T\parallel \tau(\cdot_{a_t}) - \tau(\cdot_{b_t})\parallel_1\mid \le \epsilon$ is approximately the same.
1. The reward trajectory: $\forall_{\cdot_a, \cdot_b \in D} \mid \sum_{t=0}^T \parallel r(\cdot_{a_t}) - r(\cdot_{b_t})\parallel_1 \mid \le \epsilon$ is approximately the same.

GVFs

7. The discounted future policy: $\forall_{\cdot_a, \cdot_b \in D} \mid \Pi(\cdot_a) - \Pi(\cdot_b)\mid \le \epsilon$ is approximately the same.
1. The discounted future transition: $\forall_{\cdot_a, \cdot_b \in D} \mid \Upsilon(\cdot_a) - \Upsilon (\cdot_b)\mid \le \epsilon$ is approximately the same.
1. The discounted future reward: $\forall_{\cdot_a, \cdot_b \in D} \mid Q(\cdot_a) - Q(\cdot_b)\mid \le \epsilon$ is approximately the same.

<!-- Note: two states having similar $f$ are not guaranteed to have similar abstraction! -->


__Q:__ Which is best?

> __Claim 1:__ 9.(the value fn) will yield the most compression, while performing well. But, it is a task specific representation, thus it will not transfer / generalise well.

#### Other types of abstraction

We constructed the state abstraction by altering what the policy and value function were allowed to see. Rather than observing the original state space, we gave them access to an abstracted state space.

There are other ways to alter what the policy and value function sees.

$$
\begin{align}
\phi: S \to X&: \quad \pi(s) \to \pi(\phi(s)) \quad Q(s, a) \to Q(\phi(s), a) \tag{State abstraction} \\
\psi: A\to Y&: \quad \pi(s) \to \psi^{-1}(\pi(s)) \quad Q(s, a) \to Q(s, \psi(a)) \tag{Action abstraction} \\
\phi, \psi&: \quad \pi(s) \to \psi^{-1}(\pi(\phi(s))) \quad Q(s, a) \to Q(\phi(s), \psi(a)) \tag{State and action abstraction} \\
\varphi: S \times A \to Z&: \quad \pi(s)\to \mathop{\text{argmax}}_a V(\varphi(s, a)) \quad\quad Q(s, a) \to V(\varphi(s, a)) \tag{State-action abstraction} \\
\end{align}
$$

> __Claim 2:__ The state-action abstraction is the most powerful because it allows the compression of the most symmetries. (want to prove!)


State abstraction groups together states that are similar.
For example, sprinting 100m is equivalent regardless of which track lane you are in.

Action abstraction groups together actions that are similar.
For example, X and Y both yeild the state change in state,
> Approximation perspective: we have a set of options and we want to use them to approximate the optimal policy. A good set of options can efficiently achieve an accurate approximation.

### Motivating example for state and action abstraction: ???

Might want to transfer. But some envs share state space, some share action space. Want to

- Might be teleported to a new environment? (new state space, same action space)
- Might have to drive a new vehicle (same state space, new action space)


### Motivating example for state-action abstraction: Symmetric maze
_(Some intuition behind claim 2.)_

Imagine you are in a mirror symmetric maze. It should not matter to you which side of mirror you are on.

![maze.png](../pictures/drawings/maze.png){ width=250px }

This reduces the state-action space by half! $\frac{1}{2}\mid S \mid \times \mid A \mid$. Note: just using state abstraction it is not possible to achieve this reduction. Mirrored states are not equivalent as the actions are inverted.

<!-- ## Generalised symmetries

What about other types of symmetry, other than mirror?

- $\exists f\in X: \forall_{s, a} r(s, a) = r(f(s), a)$. Where $X=GL_N \lor S_N \lor \dots$
 -->

While other learners can still solve this problem. They miss out on efficiency gains by abstracting first.


### Related work

[Near Optimal Behavior via Approximate State Abstraction](https://arxiv.org/abs/1701.04113)

## Discussion

But can we guarantee that these abstractions do not make it harder to find the optimal policy? Is that even possible?

***

Want a general way (a function) to take an abstraction of an MDP (defined by certain propreties) and return the difference between its optimal policy and the true optimal policy.
Want automated computational complexity to solve this!
Actually, we are not considering computational complexity here only approximation error.
For that can we just use automatic differentiation!?
Want a way to get bounds for all of these combinations!
