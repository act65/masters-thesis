Solving a problem via abstraction follows a generic formula. Transform the the problem into a new domain, solve the problem in this new domain, project the solution back into the original domain. In this section, we explore the way to design an abstraction so that it makes the optimisation problem 'easier'.

Firstly, what makes an optimisation problem easy? The search space is small, the search space has some structure or symmetry within it, allowing us to reduce the search space to something small(er), the search space gives us hints about how to search for what we are looking for (for example smooth and convex with respect to our loss function [convex RL](https://bodono.github.io/thesis/bod_thesis.pdf) ...), ...

And within RL, properties that would make the optimisation problem easier; a sparse transition matrix, dense rewards, linearity, ...


Note that it will not always be possible to find an efficient solution to an MDP.
Are some MDPs just fundammentally harder to solve than others?
Could mention no free lunch.

## A quick review of MDPs

The bellman equation is a non-linear optimisation problem.
Is there a way to turn this into a linear problem? What do we need to sacrifice to do so?

$$
v(s) = \mathop{\text{max}}_a \Big[r(s, a) + \gamma \mathop{\mathbb E}_{s' \sim P(\cdot | s, a)} v(s') \Big]\\
$$

^^^ What makes it non-linear?!?


## Linear markov decision problems (LMDPs)


Define an infinite horizon LMDP to be $\{S, p, q, \gamma\}$.
Where $S$ is the state space, $p: S \to \Delta(S)$ is the unconditioned transition dynamics, $q: S \to \mathbb R$ is the state reward function an $\gamma$ is the discount rate.

How can we exploit linearity to make reinforcement learning easier to understand, more efficient.

There are a few different ways we can introduce linearity to a MDP. Which one is best? We will see...

why linearity?
- it has many mathematical tools for analysis.
- we know linear systems can be solved efficiently.
- ?


Linearity is a nice property that makes optimisation simpler and more efficient.

- Linear programming (see appendix: LP)
- Linear markov decision processes

Linear optimisation is ... aka linear programming. Has a complexity of ???. Can
Solving a system of linear relationships. Has a complexity of ???.

In fact. MDPs can actually be solved via LP. see [appendix].

### Linear Markov decision process (Todorov 2009)
(Exponentiated and controlling state distributions)

How can we remove the sources of non-linearity from the bellman equation? The answer is a couple of 'tricks';

- rather than optimising in the space of actions, optimise in the space of possible transition functions.
- set the policy to be
- ?

Let's unpack these tricks and see how they can allow us to convert an MDP into a linear problem. And what the conversion costs.

$$
V(s) = \mathop{\text{max}}_u \Big[ q(s) -  \text{KL}(u(\cdot | s) \parallel p(\cdot | s)) +  \gamma \mathop{\mathbb E}_{s' \sim u(\cdot | s)} V(s') \Big]\\
$$



### Linear Markov decision process (Pires el at. 2016)
(Factored linear models)

$$
\mathcal R: \mathcal V \to W \\
\mathcal Q_a: \mathcal W \to \mathcal V^A \\
P(s'|s, a) = \int_w \mathcal Q_a(s', w)\mathcal R(w, s)\\
$$

Great. But, how does this help?

$$
\begin{align}
T_{\mathcal Q}w &= r + \gamma \mathcal Qw \\
T_{\mathcal R^A\mathcal Q}w &= \mathcal R^AT_{\mathcal Q} \\
T_{\mathcal Q \mathcal R} &= T_{\mathcal Q}\mathcal R \;\;\; (= T_P)
\end{align}
$$

It allows us to Bellman iterations in a lower dimensional space, $\mathcal W$, rather than the dimension of the transition function.

$$
\begin{align}
w^a &= T_{\mathcal R^A\mathcal Q}w \tag{bellman evaluation operator}\\
w &= M'w^a \tag{greedy update}\\
\end{align}
$$

When does this matter?
Planning!! Simulating the transition function many times. Pick $\mathcal W$...

### Linear Markov decision process (Jin el at. 2019)
()

$$
P(\cdot | s, a) = \langle\phi(s, a), \mu(\cdot) \rangle \\
r(s, a) = \langle\phi(s, a), \theta \rangle
$$


### Discussion

So which approach is best? What are the pros / cons of these linearisations?

All of them are trying to insert some kind of linearity into the transition function.

## A closer look at LMDPs

(the Todorov ones...)

A few things I want to explore;
- the composability of policies / tasks
- the embedding of actions
- LMDPs as an abstraction

Insert section on theory of LMDPs. Convergence, approx error, ...__

What are their properties?

- Efficently solvable
- Allows the composition of optimal controls.
- ???

And what are they lacking?

- Assumes we are working with a tabular representation
-

***

So now that we have explored LMDPs, how can we extract their nice properties into an architecture that might scale to more complex problems: larger state spaces and action spaces, sparse rewards, ...?

The key steps that were taken;

- Exponentiated values
- learn a policy that chooses state distributions, rather than actions.
