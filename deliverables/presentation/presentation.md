---
theme: metropolis
title: Exploration for RL
subtitle: Inductive biases in exploration strategies
author: 'Alexander Telfar'
institute: 'VUW'
date: June 30th, 2019
toc: false
slide_level: 2
header-includes: \metroset{progressbar=frametitle,sectionpage=progressbar}
---
## What is RL?

> (learning to) make optimal decisions

Context, potential actions, goal / utility function / reward.


## MDPs

$$
V(\pi) = \mathop{\mathbb E} [\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) ]
$$

$$
\pi^{* } = \mathop{\text{argmax}}_{\pi} V(\pi)
$$



## Alternative formualation

$$
\begin{aligned}
V(\pi^{* }) \equiv  \mathop{\mathbb E}_{s_0\sim d_0} \mathop{\text{max}}_{a_0} r(s_0, a_0)
+ \gamma  \mathop{\mathbb E}_{s_1\sim p(\cdot | s_0, a_0)} \Bigg[ \\ \mathop{\text{max}}_{a_1} r(s_1, a_1)
+ \gamma \mathop{\mathbb E}_{s_2\sim p(\cdot | s_1, a_1)} \bigg[ \\ \mathop{\text{max}}_{a_2} r(s_2, a_2)
+ \gamma  \mathop{\mathbb E}_{s_3\sim p(\cdot | s_2, a_2)} \Big[
\dots \Big] \bigg] \Bigg]
\end{aligned}
$$


## Why are RL problems hard?

Because of three main properties;

1. they allow, __evaluations__, but dont give 'feedback',
2. the observations are sampled __non-IID__,
3. they provide __delayed__ credit assignment.

## Example: Multi-armed Bandits

The two armed bandit is one of the simplest problems in RL.

- Arm 1: [10, -100, 0, 0, 30]
- Arm 2: [2, 0]

Which arm should you pick next?

## Why do exploration strategies matter?

Why not just do random search?

insert pic

- Too much exploration and you will take many sub optimal actions, despite knowing better.
- Too little exploration and you will take 'optimal' actions, at least you think they are optimal...

## An example: MineRL

http://minerl.io/competition/

Goal: Find and mine a diamond.

Solving this without priors is going to take a long time.

> Last time I tried to mine a yellow sparkly rock, nothing happened, this time, 1,000 actions later, I got gold. Which action(s) helped?

>


## What do we require from an exploration strategy?

- Non-zero probability of reaching all states, and trying all actions in each state.
- Converges to a uniform distribution over states. (?)
- ?

Nice to have

- Scales sub-linearly with states
- Samples states according to their variance. More variance, more samples.

What about goal conditioned exploration?

- ?

## What are some existing exploration strategies?

- Injecting noise: [Epsilon greedy](), [boltzman]()
- Optimism in the face of uncertainty
- Bayesian model uncertainty and [Thompson sampling]()
- [Counts](https://arxiv.org/abs/1703.01310) / densities and [Max entropy](https://arxiv.org/abs/1812.02690)
- Intrinsic motivation ([Surprise](https://arxiv.org/abs/1808.04355), [Reachability](https://arxiv.org/abs/1810.02274), Randomly picking goals)
- [Disagreement](https://arxiv.org/abs/1906.04161)

Note. They mostly require some form of memory and / or a model of uncertainty.
Exploration without memory is just random search...

## Counts / densities

In the simplest setting, we can just count how many times we have been in a state.
We can use this to explore states that have have low visitation counts.

$$
\begin{aligned}
P(s=s_t) = \frac{\sum_{s=s_t} 1 }{\sum_{s\in S}1} \\
a_t = \mathop{\text{argmin}}_{a} P(s=\tau(s_t, a)) \\
\end{aligned}
$$

## Intrinsic motivation

'Surprise'

$$
r_t = \parallel s_{t+1} - f_{dec}(f_{enc}(s_t, a_t)) \parallel_2^2
$$

'Reachability'

$$
r_t = \mathop{\text{min}}_{x \in M} D_k(s_t, x)
$$

## Maximum entropy

$$
\begin{aligned}
P^{\pi}(\tau | \pi) = d_0(s_0) \Pi_{t=0}^{\infty} \pi(a_t | s_t)P(s_{t+1} | s_t, a_t) \\
d^{\pi}(s, t) = \sum_{\text{all $\tau$ with $s = s_t$}}P^{\pi}(\tau | \pi) \\
d^{\pi}(s) = (1-\gamma)\sum_{t=0}^{\infty} \gamma^t d^{\pi}(s, t) \\
\pi^{* } = \mathop{\text{argmax}}_{\pi} \mathop{\mathbb E}_{s \sim d^{\pi}} [ \log d^{\pi}(s)] \\
\end{aligned}
$$

## Inductive biases in exploration strategies

So my questions are;

- do some of these exploration strategies prefer to explore certain states first?
- which inductive biases do we want in exploration strageties?
- how can we design an inductive biases to accelerate learning?
- what is the optimal set of inductive biases for certain classes of RL problem?
- how quickly does the state visitation distribution converge?

## Inductive bias

Underconstrained problems.

Occam's Razor and overfitting.

## Human bias in Minecraft

Crafting is super imporant. But has a combinatorial nature.
We bring many priors to help us. We know that;

![The various places to explore](../../pictures/images/vista.png){width=200}
![The various recipies to explore](../../pictures/images/crafting.png){width=200}

- iron is useful for making tools.
- coal and a furnace is probably needed to make iron.
- we can guess which of these is likely to be diamond
- we know that diamonds are likely to be found (deep) underground
- we

(we have an understanding of tools, and that they are the reason we got diamonds this time. This allows us to assign credit to the act of forging and mining with an iron pick-axe.)

## Implicit regularisation

Matrix factorisation ($m << d^2, Z \in \mathbb R^{d\times}$)

$$
\begin{aligned}
y_i = \langle A_i, W^{* } \rangle \\
\mathcal L(X) = \frac{1}{2} \sum_{i=1}^m (y_i - \langle A_i, XX^T \rangle )^2 \\
X^{* } = \mathop{\text{argmin}}_X \;\; \mathcal L(X)
\end{aligned}
$$

When stochastic gradient descent is used to optimise this loss (with initialisation near zero and small learning rate), the solution returned also has minimal nuclear norm $X^{* } \in \{X: \mathop{\text{argmin}}_{X\in S} \parallel X \parallel_{* } \}$,  $S =\{X: \mathcal L(X) = 0\}$.


## How do RL algorithms implicitly regularise exploration?

Surprise
- Has a bias towards states with more noise in them.

Density
- The approximation of the density may be biased.

Intrinsic motivation
- Highly dependent on its sampled history.

## The state visitation distribution

> How can we reason, in a principled manner, about bias / regularisation in exploration strategies?

$$
\begin{aligned}
d^{\mathcal A}(s, t) &= (1-\gamma)\sum_{t=0}^{t} \gamma^t Pr^{\mathcal A}(s =s_t)  \\
\end{aligned}
$$

For each different RL algol;

- Does $d(s_i, t)$ converge monotonically to $\frac{1}{n}$?
- Which $d(s_i, t)$ converge first?
- What is the difference between the $i$ different convergence rates?
- Does $d(s, t)$ converge to uniform as $t \to \infty$?

## {.standout}

Thank you!

And questions?
