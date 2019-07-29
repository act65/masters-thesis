Solving a problem via abstraction follows a generic formula. Transform the the problem into a new domain, solve the problem in this new domain, project the solution back into the original domain. In this section, we explore the way to design an abstraction so that it makes the optimisation problem 'easier'.

Firstly, what makes an optimisation problem easy? The search space is small, the search space has some structure or symmetry within it, allowing us to reduce the search space to something small(er), the search space is convex with respect to our loss function (see [convex RL](https://bodono.github.io/thesis/bod_thesis.pdf) ...), ...

And within RL, properties that would make the optimisation problem easier; a sparse transition matrix, dense rewards, ...

Linearity is a nice property that makes optimisation simpler and more efficient.

- Linear programming (see appendix: LP)
- Linear markov decision processes

Linear optimisation is ... aka linear programming. Has a complexity of ???. Can 
Solving a system of linear relationships. Has a complexity of ???.

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

There are a few different ways we can introduce linearity to a MDP. Which one is best? We will see...

(not only is linearity useful for efficient computation, it serves as a simpler setting where we can hope to gain understanding.)

- LMDPs
- LMDPs
- LMDPs

### ??? Jordan Paper

### Cepsvari paper


### Todorov: Exponentiated and controlling state distributions

How can we remove the sources of non-linearity from the bellman equation? The answer is a couple of 'tricks';

- rather than optimising in the space of actions, optimise in the space of possible transition functions.
- set the policy to be
- ?

Let's unpack these tricks and see how they can allow us to convert an MDP into a linear problem. And what the conversion costs.


$$
v(s) = \mathop{\text{max}}_u \Big[r(s) + \gamma \mathop{\mathbb E}_{s' \sim u(\cdot | s)} v(s') \Big] \\
\text{s.t.}  \;\; u(\cdot | s) \;\;\; ??? \\
$$


$$
l(s, a) = q(s) + KL(u(\cdot | s) \parallel p(\cdot | s)) \\
v(x) = q(s) + \mathop{\text{max}}_a \Big[ KL(u(\cdot | s) \parallel p(\cdot | s)) +  \gamma \mathop{\mathbb E}_{x' \sim P(\cdot | x, a)} v(x') \Big]\\
$$

### Discussion

So which approach is best? What are the pros / cons of these linearisations?



## A closer look at LMDPs

(the Todorov ones...)

A few things I want to explore;
- the composability of policies / tasks
- the embedding of actions
- LMDPs as an abstraction

Insert section on theory of LMDPs. Convergence, approx error, ...__
