## Value function and critics

How can we estimate the dreivative of a stochastic, unknown and possibly discrete function?
An answer is to learn a critic, a differentiable approximation of the stochastic, unknown function.

__Q:__ Which function approximators are suited to the types of function we are interested in approximating (changing distribution, sparse/unbalanced, ...?)

### REINFORCE

$$
\begin{align*}
L(\pi) &= \mathbb E_{s\sim\pi}[R(s)] \\
\pi^* &= \mathop{\text{argmax}}_{\pi}  L(\pi) \\
\nabla L(\pi) &= \nabla \mathbb E_{s\sim\pi}[R(s)] \\
&= \nabla_\theta \int \pi(s, \theta) R(s) ds\\
&=  \int \nabla_\theta \pi(s, \theta) R(s)ds\\
&=  \int \frac{\pi(s, \theta)}{\pi(s, \theta)}\nabla_\theta \pi(s, \theta) R(s)ds\\
&=  \int \pi(s, \theta)\nabla_\theta \log\pi(s, \theta) R(s)ds\tag{$\nabla_x \log(z(x)) = \frac{1}{z}\nabla_x z(x)$}\\
&= E_{s\sim \pi}[-\nabla_\theta log(\pi(s, \theta)) R(s)]\\
\end{align*}
$$

Advantage actor critic improves this by reducing the variance of the gradient estimation.

$$
\begin{align*}
A &= R(s, a) - V(s) \\
&= Q(s, a) - V(s) \\
&\implies -\nabla log(\pi(s)) A \\
\end{align*}
$$

__Q:__ Why does less variance mean faster learning!?

But what if $V(s)$ is not a reliable estimate of $R(s)$? Are there cases where this could actually give worse behaviour? How about the average case in training?

Hypothesis: because we are using a neural network to estimate $V(s)$, when $r(s)$ is sufficiently sparse, then the neural net will collapse to a near constant function.
Meaning it provides little variance reduction.

### Resources

- [REINFORCE](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
- [A3C](https://arxiv.org/abs/1602.01783)
- [Generalised advantage estimation](https://arxiv.org/pdf/1506.02438.pdf)
- [Distributional RL](https://arxiv.org/abs/1806.06923)
- [Backprop through the void](https://arxiv.org/abs/1711.00123)

## Model-based RL

__Cojecture:__ Model-based learning is the optimal solution to model-free learning. Aka the Q network will learn to model the dynamics of the environment.
(optimal if you include computatunal complexity and ability to transfer to new tasks)

How hard is it for a model-free system (to learn) to learn to use future inputs as targets for training?

Model-free methods must learn someweak correlate of dynamics!? They must know which states are reachable to give valid estimates.

## Resources

- [TDM](https://bair.berkeley.edu/blog/2018/04/26/tdm/)
- [Successor features](https://arxiv.org/abs/1606.05312)
- [Model-free planning](https://openreview.net/forum?id)

## Planning

CFRM, MCTS, ... cool.
What about planning in with continuious actions?  -> LQR, MPC

__Q:__ How much harder is planning in a stochastic system than in a deterministic system?!?

### Model predictive control

(what about LQR, ...)

Short coming of MPC. Finite horizon.
- Will be short sighted/myopic? will fail at problems with small short term reward in wrong direction?
- Cannot trade off width for depth.
- What if the state returned by the model is a distribution? Then we need to explore possibilities!?!?

Can derive Kalman filters from these?!!

$$
\begin{align}
V(x) = min_u [l(x, u) + V(f(x, u))] \tag{the cts bellman eqn!?}\\
\end{align}
$$

> The Q-function is the discrete-time analogue of the Hamiltonian (!?!?)

Planning/Reasoning with limited compute. Only get to evaluate N trajectories. Looking for a single trajectory that is sufficient. Versus the best possible trajectory1?

### General problem

Need to integrate a dynamical system.
But how to do this when it is;
- stochastic?
- biased?

Want to learn to integrate!?

### Resources

- [Differential Dynamic Programming](https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf)
- [mpc.torch](https://locuslab.github.io/mpc.pytorch/)
- ?


## Search

__Q__ Efficient search. Relationship between tree search and gradient descent? Local and global optimisation.


Why do we want to do this?
Optimisation in the loop.

### Definition

Searching through policy space.
Searching through parameter space.
Searching through state space(the env).

Making the most of self-play.


### Idea - searching through param vs action

At the end of the day we are searching for a policy, what action to take when. We are searching through distributions over state-action pairs.
But instead, we decide to frame this as searching through parameters of a NN that pairs the state-actions.

The problem is that often changes in NN params my have little/no effect on the state-action pairs. Therefore serching there is a waste of time.

So we should scale the gradients by the sensitivity of the actions to the parameters!?

Moving through distribution space versus moving through parameter space. How are they structured? What are the differences?


$$
\frac{\partial a}{\partial \theta}^{-1}\cdot\frac{\partial L}{\partial \theta}
$$

## Invariance under a contraction operator

 Both Bellman and GD, and !?.

$$
\begin{align}
x(t+1) &= T_{GD} x(t) \\
&= x(t) - \eta \frac{\partial E}{\partial x} \\
x_i(t+1) &= T_{BM} x_i(t) \\
&= R + \gamma \mathop{\text{max}}_a \int p(x_i(t), a) \cdot x_{i+1}(t) dx  \\
\end{align}
$$

This just means the steps/iterations will converge to a fixed value.

Hmph.
- But GD isnt always a contraction? GD is only a contraction operator when the loss surface is convex?
- My formulation of the bellman operator doesnt seem right. Is the bellman operator a contraction over $t$ or $i$, or both?

Is it possible to do an eigen analysis of these linear operators?

## Representation

What if I have an independent variable that is a ring, or has some other topology other than a line?

The latent variable to match the generating variable types. (the ability to construct sets equipped with a metric/product/transform!? mnist -> a dimension with 10 categorical variables, a ring of reals describing azimuth, bounded dims describing translation, ...)

What about a dataset with varying numbers of sprites/digits in the image.

## Transition fn

Want a linear decomposition of the transition function so we can use feature expectations.

$$
\begin{align}
s_{t+1}^i = f(s_t^i) \tag{wwhitney} \\
s_{t+1}^i = s_t^i +  \\
\end{align}
$$

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

***

- __Q__ What can you learn from an interactive environment (you can take actions and observe their results) that you cannot learn from a (possibly comprehensive) static dataset? Similarly, what can you learn when you have access to an $\epsilon$-accurate model, that you cannot learn from an interactive environment?
- (recomposable) modular systems. Want to be able to build up complexity from simple parts. Recursively!
- symmetric transformations/factorisation of the game tree. learn f(s) such that the resulting game tree is the same!?
- Distributed representations (various tensors) don't store knowledge in nice ways... What alternative representation are there?
- Relationship to bases. Is there a way to reason about a basis with many different ways of combining the bases? More complicated structure? (designing algebras!?)
- learning the long term effects of actions OR exploration!? OR unsupervised tasks/learning from context/automatic curriculum/? OR using temporal info to disentangle?
