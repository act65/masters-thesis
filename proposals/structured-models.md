% structure
# Structured models

Why is this hard?

- uncertainty
- partial observations
- structure
- complex
- ?

## Definition

Want the model to approximate the true internal structure of the environment, not just an arbitrary structured function approimation.
Want to recover the truth. Do science, extract relationships, reduce into minimal parts.

## Partial observations

How can we construct global models from local observations?

__Conjecture:__ To build a global model is it necessary for the agent to estimate its poisition in its environment.

But where does the ground truth signal come from!?

## Modelling uncertainty

Distributions over plausible states?
MCMC estimation versus using flows?

## Graph neural networks

!?!?
How to contruct a structured representation from partial observations!?!!

## Approximate gradient dynamics

Want to recover the dynamics of a system, and its state. How can these be disentangled??

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



## Causal inference

???

## Compression

???

## Resources

- []()
- []()
