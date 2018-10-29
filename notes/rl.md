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

### Approximate gradient dynamics

How can we efficiently approimate a dynamical system?

We have observations $\{x(t_0),x(t_1), \dots , x(t), x(t_{+1}) \}$ and know that they are produced via a dynamical system.

$$
\begin{align*}
\frac{dx}{dt} &= f(x) \\
\end{align*}
$$

__NOTE__ Would make it easy(er) to design an exploitable model!?
Controller simply inputs stepsize.

### Inverse gradient dynamics
(relationship to IRL?)

Assume that the dynamcs of a system are the result of an energy being minimised.

$$
\begin{align*}
x_{t+1} &= x_t - \eta\nabla E(x_t) \tag{the true dynamics}\\
f(x_t) &= x_t - \nabla \hat E_{\theta}(x_t) \tag{approximate the energy fn}\\
L &= \mathop{\mathbb E}_{x_{t+1}, x_t\sim D} \big[ d(x_{t+1}, f(x_t)) \big] \tag{loss fn for approximation}\\
\end{align*}
$$

- (how does this constrain the possible types of dynamics?)
- if we use an ensemble of approximators, how will the dynamics be decomposed? will we naturally see environment disentangled from agents?


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

### Inversion/Backwards/Abduction

What are the advantages of having access to a inverse dynamics model?

- Some problems are easier to solve? (U-problem?)
- Smaller search space (1 vs 2 circles?)
- ?

***

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


## Transfer

Why do we want to do this? Transfer learning is the key to general intelligence!

### Definition

What do we mean by "transfer learning"? If we have two tasks/environments/action spaces/...?, $A, B$, then the performance of one task aids the other task.

A MDP is defined as
$$M = \Big(S, \mathcal A, p(\cdot \mid s,a), R(s, a, s') \Big)$$

- $S$: It is possible to change the state space, while preserving the dynamics. (??)
- $\mathcal A$: Change the action space, for example, instead of $\leftarrow, \rightarrow, \uparrow, \downarrow$ we use $\uparrow, \text{rot90}$
- $p(\cdot \mid s,a)$: from subtle things like not being able to reach a state on another one, to chan
- $R(s, a, s')$: A different reward funciton, aka a different task.

But one could imagine symmetries of $p(\cdot \mid s,a), R(s, a, s')$, such that some structure is preserved.

$$
\begin{align*}
p(\cdot \mid s,a) &:= T^{-1}(p(\cdot \mid T(s,a))) \\
&:= p(\cdot \mid T(s),a) \tag{equiv to transfer to a new state space}\\
&:= p(\cdot \mid s,T(a)) \\
R(s, a, s') &= T(R(s, a, s')) \\
\end{align*}
$$

For example, similarities between the reward in hockey and football. Get the round thing in the oppositions goal.

> Huh, never thought about it this way before. The states are an unordered set.
The transition fn provides all the structure on that space (much like an inner prod in Hilbert spaces?!?)
The neighbors of a state are the positions reachable from a single action.
No not quite. More like probabilistic vector maps? No that is only when combined with a policy.

Best current solutions!?

- successor representation/goal embeddings. $\to$ task transfer
- model-based RL (disentangle policy from model) allows transfer of control polices between environments and transfer of model between tasks in the same env.

Let $L$ be the test loss after training, and $T$ be the training task.
$$
\begin{align*}
L(T(B)) \le L(T(A) \to T(B)) \tag{Forward transfer}\\
L(T(A)) \le L(T(A) \to T(B)) \tag{Backward transfer} \\
\end{align*}
$$

Relationship to meta-learning. Different <i>'levels'</i> of knowledge can be transfered. In meta learning the low level details are not transferred, but the high level, "how to learn" lessons are transferred. So the key to this would be a decomposition of these different types of knowledge. __Q:__ How can these types of knowledge be disentangled!?

$$
\begin{align*}
\dot L(T(B)) \le \dot L(T(A) \to T(B)) \\
\dot L(T(A)) \le \dot L(T(A) \to T(B)) \\
\end{align*}
$$

### Analysis

What I would really like is a set of tools for analysing transfer learning.
I would like to be able to answer the questions;

- what knowledge was transferred (high level, low level, ...?)
- how was it transferred? (if we are dealing with NNs then how does some knowledge get shared while other knowledge doesnt?
because the existing knowledge allows faster learning?!)
- why was it transferred? (because the domains somehow shared similarities)

Seems quite related to representation learning. The key will be how knowledge is represented, and how easily that knowledge can be translated (/transformed)!?


If we had a theory of transfer learning we would be able to;
- predict when X will transfer to Y.
- write down a pattern to generate representations for transfer between X/Y.
- __???__

### Toy problems

Want to generate different MDPs that share various 'orders' of similarity.
If we mode each environment as a graph and the task is navigation, then it might be possible to easily generate graphs/rewards with structural similarities!? Various orders of persistent homology?

## Long term credit assignment

!!!??!?!??!?
Rudder!


## Compression

- MLD? symmetry strucutre?  What are the 'right' priors? How can we optimise them?

***

- __Q__ What can you learn from an interactive environment (you can take actions and observe their results) that you cannot learn from a (possibly comprehensive) static dataset? Similarly, what can you learn when you have access to an $\epsilon$-accurate model, that you cannot learn from an interactive environment?
- (recomposable) modular systems. Want to be able to build up complexity from simple parts. Recursively!
- symmetric transformations/factorisation of the game tree. learn f(s) such that the resulting game tree is the same!?
- Distributed representations (various tensors) don't store knowledge in nice ways... What alternative representation are there?
- Relationship to bases. Is there a way to reason about a basis with many different ways of combining the bases? More complicated structure? (designing algebras!?)
- learning the long term effects of actions OR exploration!? OR unsupervised tasks/learning from context/automatic curriculum/? OR using temporal info to disentangle?
