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

### Completeness

(_want to show that the way we build our repesentation is capable of capturing all symmetries_)

$$
\begin{align}
\text{Pick} \; (s, a, f, g) \; \text{s.t.}\; s' = f(a), a' = g_s(a) \\
???
\end{align}
$$


### Related work

##### Symmetry learning for function approximation in RL
(https://arxiv.org/abs/1706.02999)

They use reward trajectories to construct a notion of similarity between two state-action pairs.

This technique allows one-shot generalisation to new state-action pairs. A new state-action pair is observed, and it has the sample reward trajectory as another state-action pair. We can automatically transfer the value estimate from the former pair to the latter. (for the current policy...).
__Actually. No?!!?__

Depending on how much symmetry is displayed, we are increasing the amount of data each state - action pair has...? How much more data per symmetry?
A form a weight sharing!
This is thought to be why conv nets work so well, their kernels recieve huge amounts of data as they are shared over many spatial locations.

Problems.
- It requires discrete state / action spaces, as to represent the ??? they use a tree.
- It doesnt scale well. Number of possible future trajectories = ???.
- Requires dense reward

$N: [n_{states}\times n_{actions} \times l_{steps} \times k_{possibilities}]$

$$
\begin{align}
N(s, a, l, k) &= ??? \\
\sum_{l, k} &= \sum_{l=l_o}^L \sum_{k_o}^K \\
\chi(\langle s,a\rangle, \langle s', a'\rangle) &= \frac{\sum_{l, k} \text{min} (N(s, a, l, k), N(s', a', l, k))}{\sum_{l, k} N(s, a, l, k) \sum_{l, k} N(s', a', l, k))} \\
\chi(\langle s,a\rangle, \langle s', a'\rangle) &= \frac{\sum_{l, k} p(\tau^k_l | s, a) p(\tau^k_l | s', a')}{\sum_{l, k} p(\tau^k_l | s, a) \sum_{l, k} p(\tau^k_l | s', a'))} \\
\chi(\langle s,a\rangle, \langle s', a'\rangle) &= JS(p(\cdot| s, a) \parallel p(\cdot | s', a'))\\
\end{align}
$$

^^^ This reminds me of testing for statistical independence!? $1 = \frac{P(A,B)}{P(A) P(B)}$

The estimates of $D(p(\cdot | s, a), p(\cdot | s', a'))$ are independent of policy. Sure, the current policy will effect the distribution of trajectories $p(\cdot | s, a)$. But if $\exists f, g: s' = f(s), a' = g_s(a)$ then $p(\cdot | s', a')$ will be effected equally.
But, if $p(\cdot | s, a)$ changes allot, then we need to explore enough to also update $p(\cdot | s', a')$. Hmm. Would rather couple the two?! $p(\cdot | s, a) = p(\cdot | s', a')$ once we have figured out that they are 'similar' (under a stationary policy). Want to generalise to other policies.

$$
 \chi(\langle s,a\rangle, \langle s', a'\rangle) >\Delta \\ \implies \mathop{\text{min}}_{\theta} D\big( \zeta(s, a), \zeta('s, a')\big) \\
\mathop{\text{min}}_{\theta} \mathop{\mathbb E}_{\chi} \big[\parallel \zeta(s, a) - \zeta(s', a') \parallel_2^2 \big]\\
$$

- Problem. We have gained data efficiency, but not computational effeciency? We need to train the network for each of these symmetries.
- Question. If we are training a NN in this way, how does the invariance get implemented within the NN?
- As training proceeds, and more symmetries have been observed. There might be very many pairs that are 'similar'. Want to visualise these clusters?!
- Oh... All we are doing is clustering based on a similarity measure... How does that relate to symmetry and quotients?

##### Near Optimal Behavior via Approximate State Abstraction
(https://arxiv.org/abs/1701.04113)

### Notes

Struggling with the direction of implication, $\phi (s_1) = \phi(s_2) \implies \forall_a \mid Q(s_1, a) - Q(s_2, a)\mid \le \epsilon$, what about $\phi (s_1) = \phi(s_2) \impliedby \forall_a \mid Q(s_1, a) - Q(s_2, a)\mid \le \epsilon$?

But can we guarantee that these abstractions do not make it harder to find the optimal policy? Is that even possible?

So this is about messing with what information the value function has. Given more information about the future, we should expect the acuracy (or speed of learning) of the estimate to go up!?

Want a general way (a function) to take an abstraction of an MDP (defined by certain propreties) and return the difference between its optimal policy and the true optimal policy.

Want automated computational complexity to solve this!


### Disentangled action abstractions

(what can we prove about this!?)
How does this related to finding symmetries and state-actions!?
