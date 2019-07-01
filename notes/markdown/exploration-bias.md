Inductive bias in exploration.

Different exploration strategies might prefer to explore some areas before others.

How can we quantify an exploration strategy?

- calculate the steady-state distribution over states. No.

the important point is the order of exploration.
- first need to show that one exploration stragety has a prefered ordering.
- then can compare different exploration strategies.

$$
P(s, t) = d^{\pi} [s = s_t]
$$

__Observation:__ If for all t, the entropy is maximised, then there is no bias.

$$
\forall t, H(P(s, t)) = 1/n
$$

__QUESTION:__ Do some exploration strategies have a bias in the limit of $t\to \infty$?

$$
H(P(s, \infty)) \neq 1/n
$$

For any exploration strategy, a requirement should be that it conveges to the max entropy state-distribution.


***

Wait, what are we exploring?
Policies? Or states?
States.
We definitely do not want to explore all possible policies...

***

What priors could make sense in exploration?

Want to explore first;
- states that we can control,
- states that allow us to control many other dimensions of the state
- states that behave the most unpredictably (have large variance).
  or the opposite? states that we can be certain about.
- state-actions with large effect. a large change in state
- actions with disentangled effects (leaving other state dimension invariant)
- states / actions that allow efficient traversals. (fewer actions required to get from A to B)

***

How does this make sense?
- in discrete spaces. likely to have a bias towards more central nodes? the ones that are 'easy' to reach.
- how does inductive bias make sense in cts spaces? exploring some dimensions more than others. but we can find another basis when that might not be true?!

***

Surprise has the inductive bias that it wants to explore noisy states first. The nosier the better, as they are the least predictable.
This doesnt seem like a useful inductive bias...


***

$$
\pi_0 = \mathcal A(d_0) \\
\pi_1 = \mathcal A(d_{\pi_0}) \\
\pi_2 = \mathcal A(d_{\pi_1}) \\
$$

$$
\begin{align}
\frac{d\pi}{dt} &= \mathcal A(d(\pi(t))) \\
&= \nabla_{\pi} H(d(\pi(t))) \tag{max entropy}\\
\end{align}
$$

__Assumption:__ Between each time step there have been enough samples to accurately estimate the state visitation distribution $d(\pi)$. (_This might be an ok assumption to start with_)

$$
\frac{\partial d}{\partial t} = \frac{\partial d}{\partial\pi} \frac{\partial\pi}{\partial t} \\
\epsilon(t) = D(d(t), ?)
$$


***

exploration strategies

- optimisim in the face of uncertainty
- intrinsic motivation
  - surprise (rewarded for seeing unpredictable states)
  - novelty (rewarded for seeing new states)
- count based
  - and pseudo count?
- max entropy

***

I expect that intrinsic motivation exploration strategies will be highly dependent on their past.
If it sees a few rewards for doing X, then it will continue to explore within X, possibly getting more rewards.
Positive feedback.

Is also dependent on how the value fn approximator generalises the rewards it has seen.


***

$$
\begin{aligned}
P^{\pi}(\tau | \pi) = d_0(s_0) \Pi_{t=0}^{\infty} \pi(a_t | s_t)P(s_{t+1} | s_t, a_t) \\
d^{\pi}(s, t) = \sum_{\text{all $\tau$ with $s = s_t$}}P^{\pi}(\tau | \pi) \\
d^{\pi}(s) = (1-\gamma)\sum_{t=0}^{\infty} \gamma^t d^{\pi}(s, t) \\
\end{aligned}
$$

Does this discounted state distribution really make sense???

Convergence
$$
KL(d^{\pi}(s, t), d^{\pi}(s))
$$


***

- note. but we dont just care about exploring states??? inductive bias in state-actions?



***

Relationship to the heat equation?!
$$
\frac{\partial d}{\partial t} = \alpha \frac{\partial^2 d}{\partial s^2}
$$

***

Want: for a given finite time horizon, the state visitation distribution is approimately max entropy. If we only require convergence in the limit, we could

Also, algorithms with short memories may forget.
