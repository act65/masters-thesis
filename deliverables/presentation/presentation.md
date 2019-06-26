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

Reinforcement learning is a (sub)set of solutions to the collection of optimal control problems the look like;

$$
V(\pi) = \mathop{\mathbb E} [\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) ]
$$

$$
\pi^{* } = \mathop{\text{argmax}}_{\pi} V(\pi)
$$

## ???

$$
\begin{aligned}
V(\pi^{* }) =  \mathop{\mathbb E}_{s_0\sim d_0} \mathop{\text{max}}_{a_0} r(s_0, a_0)
+ \gamma  \mathop{\mathbb E}_{s_1\sim p(\cdot | s_0, a_0)} \Bigg[ \\ \mathop{\text{max}}_{a_1} r(s_1, a_1)
+ \gamma \mathop{\mathbb E}_{s_2\sim p(\cdot | s_1, a_1)} \bigg[ \\ \mathop{\text{max}}_{a_2} r(s_2, a_2)
+ \gamma  \mathop{\mathbb E}_{s_3\sim p(\cdot | s_2, a_2)} \Big[
\dots \Big] \bigg] \Bigg]
\end{aligned}
$$


## Why are RL problems hard?

Some of the following properties;

1. allow, evaluations, but dont give 'feedback',
2. the data is not sampled IID,
3. provide delayed credit assignment.

## Example: Multi-armed Bandits

The two armed bandit is one of the simplest problems in RL.

1. [10, -100, 0, 0, 0]
2. [2, 0]

Which arm should I pick next?

## Why do exploration strategies matter?

Why not just do random search?

insert pic

- Too much exploration and you will take many sub optimal actions, despite knowing better.
- Too little exploration and you will take 'optimal' actions, at least you think they are optimal...

## An example: Minecraft!

Crafting is super imporant. But has a combinatorial nature.
We bring many priors to help us. We know that;

- iron is useful for making tools.
- coal and a furnace is probably needed to make iron.
- ?

![The various places to explore](../../pictures/images/vista.png){width=200}
![The various recipies to explore](../../pictures/images/crafting.png){width=200}

## What is an inductive bias?

Underconstrained problems.

Why might this matter in exploration?

## Example: Matrix factorisation

Lowest rank solution

- wug test?



## What do we require from an exploration strategy?

- Non-zero probability of reaching all state, and trying all actions in each state.
- Converges to a uniform distribution over states. (?)
- ?

Nice to have

- Scales sub-linearly with states
- ?

## What are some existing exploration strategies?

- Injecting noise: [Epsilon greedy](), [boltzman]()
- Optimism in the face of uncertainty
- [Thompson sampling]()
- [Counts](https://arxiv.org/abs/1703.01310) / densities
- Intrinsic motivation ([Surprise](https://arxiv.org/abs/1808.04355) and [Reachability](https://arxiv.org/abs/1810.02274))
- [Max entropy](https://arxiv.org/abs/1812.02690)
- [Disagreement](https://arxiv.org/abs/1906.04161)
- Randomly picking goals

Note. They mostly require some form of memory.
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

## Intrisnic motivation

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

## A principled approach.

> How can we reason about inductive biases in exploration strategies in principled manner?

Convergence
$$
KL(d^{\pi}(s, t), d^{\pi}(s))
$$

## {.standout}

Thank you!

And questions?
