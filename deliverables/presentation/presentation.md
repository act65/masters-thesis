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

Reinforcement learning is a collection of solutions to problems that;

- allow, evaluations, no feedback,
- have delayed credit assignment. (usually, but not necessarily)

## Example: Bandits

The two armed bandit is the simplest problem in RL.

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

## What do we require from an exploration strategy?

- Non-zero probability of reaching all state, and trying all actions in each state.
- Converges to a uniform distribution over states. (?)
- ?

Nice to have

- Scales sub-linearly with states
- ?

## What are some existing exploration strategies?

- Counts / densities
- Intrinsic motivation (Surprise and Novelty)
- Optimism in the face of uncertainty
- Max entropy

Note. They all require some form of memory.
Exploration without memory is just random search...

## Counts / densities

In the simplest setting, we can just count how many times we have been in a state.
We can use this to pick states that have have low visitation counts.

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

'Novelty'

$$
r_t = \mathop{\text{min}}_{x \in M} D(s_t, x)
$$

## What is an inductive bias?

Underconstrained problems.

Why might this matter in exploration?

## Example: Matrix factorisation

Lowest rank solution

- wug test?

## Inductive biases in exploration strategies

So my questions are;

- do some of these exploration strategies prefer to explore certain states first?
- which inductive biases do we want in exploration strageties?
- how can we design an inductive biases to accelerate learning?
- what is the optimal set of inductive biases for certain classes of RL problem?

## {.standout}

Thank you!

And questions?
