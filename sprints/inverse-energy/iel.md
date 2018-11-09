# IEL

## Related work

What is the difference between learning a distribution and learning an energy?!?
Obviously, the distribution needs to be normalised. But is that the only difference?

$$
\begin{align}
\frac{ds}{dt} &= -\eta\frac{\partial p(s)}{\partial s} \\
\frac{ds}{dt} &= -\eta\frac{\partial E(s)}{\partial s} \\
\end{align}
$$

#### Relationship to energy-based ML?

Energy-based ML use a fixed measure of energy, for example, RBMs:

$$
E(u) =-\frac{1}{2} \sum_{i}\sum_{j\neq i} W_{ij}\rho(u_i)\rho(u_j) - \sum_i b_i\rho(u_i)\\
$$

And adapt the connectivity/topology of the network.

#### How is this idea different from related work?

Energy based learning seems to focus on matching inputs and outputs (for classification or ...), not the internal dynamics.


### Alternative formalisations

Let $E$ be the energy function being minimised, and let $x_t$ be our observations (and/or the state) at time $t$.

What are the different contractive operators?

$$
\begin{align}
x_{t+1} &= f(x_t) \\
x_{t+1} &= x_t - \eta \nabla E(x_t) \\
\end{align}
$$

OR

$$
\begin{align}
x &\sim p(x) =\frac{e^{-E(x)}}{\sum_j e^{-E(x_i)}}
\end{align}
$$

How are these related!? Both will find the minima. (?).

OR

$$
\begin{align}
x_{t+1} &= x_t \cdot \frac{e^{\nabla E(x_t)}}{Z} \\
\end{align}
$$

Is this equivalent!?

## Setting

Given many different observations. We are able to _explain_$^{* }$ them using a simple energy function (or principle).

We receive a set of observations $\{x_0, \dots, x_n \in X\}$ that are the result of some known dynamical system $\frac{dx}{dt} = f(x_t) + y$. Our goal is to learn an energy fuction $E(x, y)$ and a set of initial conditions $\{z_0, \dots, z_n\}$ such that $x_0 \mathop{\text{min}}_x E(x, y) \text{ s.t. init } x=z_0$ (the steps taken dont need to be GD steps of down E).

Culd rewrite as $E(x^S)$. We are not allowed to control the entire state. Only part of it. We are

So the energy function needs to learn to pick out some structure in the current state that is being minimised by $f$.

> relationship to MDL!?

- https://arxiv.org/abs/1805.09714
- https://arxiv.org/abs/1810.10525

## Gradients

Ok, what do the gradients of IEL look like?

$$
\begin{align}
f_{\theta}(x_t) &= x_t - \eta \frac{\partial E_{\theta}}{\partial x} \tag{by defn}\\
\frac{\partial f_{\theta}}{\partial \theta} &= -\eta\frac{\partial^2 E_{\theta}}{\partial x \partial \theta} \tag{diff wrt params}\\
\frac{\partial L}{\partial \theta} &= \frac{\partial L}{\partial f_{\theta}} \cdot \frac{\partial f_{\theta}}{\partial \theta} \tag{chain rule}\\
&=  \frac{\partial L}{\partial f_{\theta}} \cdot -\eta\frac{\partial^2 E_{\theta}}{\partial x \partial \theta} \tag{substitution}\\
\frac{d \theta}{d t} &= -\alpha\frac{\partial L}{\partial \theta} \tag{GD}\\
&=  \alpha\eta \frac{\partial L}{\partial f_{\theta}} \cdot \frac{\partial^2 E_{\theta}}{\partial x \partial \theta} \tag{substitution}\\
\end{align}
$$

__BUG__ Third last and last line are wrong. The parameters are used multiple times over the inner GD iterations. Will need something like BPTT. True for a single step tho?

How are the dynamics of training with this update different from GD?

Linear layers behave differently?

## Densities

Assume the data you observe $\{\{x^0_i, \dots, x^t_i\}: i\in [1:N] \}$ are the result of Langevin dynamics on a probability density.

$$
\begin{align}
x(t+dt) &= x + \frac{\alpha}{2} \nabla_x p(x_t) + z \tag{$z\sim N(0, \alpha)$}\\
p(x_{t+1}) &= p(x_t) -\eta \nabla_x D (p(x_t), \pi(x_t)) \tag{!?} \\
\end{align}
$$

Langevin relationship between optimisation and sampling!? [welling](https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf)

Which is more general? Energy or density?

### Refs

- [OpenAI EBM](https://arxiv.org/abs/1811.02486)
- [EBMs review](https://arxiv.org/abs/1708.06008)
- [GANs, IRL, EBMs](https://arxiv.org/abs/1611.03852)
- [Tutorial](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)
- [BM for time-series](https://arxiv.org/pdf/1708.06004.pdf)
