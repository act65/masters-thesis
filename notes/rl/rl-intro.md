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

### Resources

- [Differential Dynamic Programming](https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf)
- [mpc.torch](https://locuslab.github.io/mpc.pytorch/)
- ?


***

Statistical estimation problem.
Ok, so you start off being uncertain about everything.
But then you see some data and can estimate the value of some actions.
You then use these estimates to change your policy.

But those estimates might not be accurate.
And changing your policy might mean that it takes you longer to realise that your evaluation of those states was very wrong.

***

Problem that occurs in POMDPs. Am I correctly modelling the state?
You want to learn what action a does. So you do $\tau(s, a)$ over many $s\in A$. But the effect of $a$ correlates with the subset of states $A \subset S$ yo are experimenting in.
For example. Balls always fall towards the gound (if you test only on earth).

***

- [ ] Equivalence of goal/option conditioned value fns
- [ ] Build a three (or even better, N) layer heirarchy
- [ ] Explore how different approaches scale (computational complexity) in the number of layers in the heirarchy
- [ ] A heirarchical subgoal net that uses MPC rather than learned policies
- [ ] Explore function approximation for options a = f(w) (rather than look up table)
- [ ] How to achieve stable training of a hierarchy?
- [ ] The benefit of a heirarchy of abstractions? (versus use a single layer of abstraction). Transfer!?
- [ ] Design a new programming language. Learner gets access to assembly and must ??? (build a calculator? allow a learner to build websites easy? ...?). What would be the task / reward fn? (should be easy to learn to use, require few commands to do X, ...?)
- [ ] A single dict with the ability to merge, versus a heirarchy!? Or are they the same?

> 1) What do we mean my abstraction? Let's generate some RL problems that can be exploited by a learner that abstracts.

> 2) How does abstraction actually help? Computational complexity, sample complexity, ... when doesn't it help? When it is guaranteed to help?

> 3) Can we learn a set of disentangled actions. How does that help?

> 4) How can we use an abstraction to solve a problem more efficiently? Use MPC + abstraction. Explore how different abstractions help find solutions!?

> 5.Build a differentiable neural computer (Graves et al. 2016) with locally structured memory (start with 1d and then generalise to higher dimensions). Is the ability to localise oneself necessary to efficiently solve partial information decision problems? Under which conditions does the learned index to a locally structured memory approximate the position of the agent in its environment.


Memory structures + Locality.

- https://www.nature.com/articles/nn.4661.pdf
- https://arxiv.org/pdf/1602.03218.pdf
- https://arxiv.org/pdf/1609.01704.pdf
- DNC
