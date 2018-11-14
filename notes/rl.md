## Value function and critics

How can we estimate the dreivative of a stochastic, unknown and possibly discrete function?
An answer is to learn a critic, a differentiable approximation of the stochastic, unknown function.

- Which function approximators are suited to the types of function we are interested in approximating (changing distribution, sparse/unbalanced, ...?)
- ??

### REINFORCE

$$
\begin{align*}
L(\pi) &= \mathbb E_{s\sim\pi}[R(s)] \\
\pi^* &= \mathop{\text{argmax}}_{\pi}  L(\pi) \\
\nabla L(\pi) &= \nabla \mathbb E_{s\sim\pi}[R(s)] \\
&= \nabla \int \pi(s) R(s) \\
&= -\nabla log(\pi(s)) R \\
\end{align*}
$$


(why does less variance mean faster learning!? need to motivate)
Advantage actor critic improves this by reducing the variance of the gradient estimation.

$$
\begin{align*}
A &= R(s) - V(s) \\
&\approx V(s_t) + r(s) - V(s) \\
&= -\nabla log(\pi(s)) A \\
\end{align*}
$$

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

(In the unconstrained memory case)
__Cojecture:__ Model-based learning is the optimal solution to model-free learning

I can imagine a model-free system learning to learn to use future inputs as targets to learn a model!!?!
If we used a very large RNN to model $Q(s_t, a_t)$, it could correlate the difference between past and future state and actions, thus ...?

Model-free methods must learn some estimate of future dynamics!? (how can this be shown? neurons that correlate with dynamics?)
How much of the future dynamics does a model-free method learn? How much is necessary?

## Resources

- [TDM](https://bair.berkeley.edu/blog/2018/04/26/tdm/)
- [Successor features](https://arxiv.org/abs/1606.05312)
- [Model-free planning](https://openreview.net/forum?id)

## Structured models

Why is it hard to learn a good model?

- uncertainty
- partial observations
- complexity
- low frequency events (how is this related to complexity?)

### Causal inference

- How is RL like the interventional level of the causal heirarchy?
- How is model-based RL like the counter-factual level of the causal heirarchy?
- How can we automate reductionism?
- ?



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


***

- __Q__ What can you learn from an interactive environment (you can take actions and observe their results) that you cannot learn from a (possibly comprehensive) static dataset? Similarly, what can you learn when you have access to an $\epsilon$-accurate model, that you cannot learn from an interactive environment?
- (recomposable) modular systems. Want to be able to build up complexity from simple parts. Recursively!
- symmetric transformations/factorisation of the game tree. learn f(s) such that the resulting game tree is the same!?
- Distributed representations (various tensors) don't store knowledge in nice ways... What alternative representation are there?
- Relationship to bases. Is there a way to reason about a basis with many different ways of combining the bases? More complicated structure? (designing algebras!?)
- learning the long term effects of actions OR exploration!? OR unsupervised tasks/learning from context/automatic curriculum/? OR using temporal info to disentangle?
