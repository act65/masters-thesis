### Gradients

Ok, what do the gradients of IEL look like?

$$
\begin{align}
f_{\theta}(x_t) &= x_t - \eta \frac{\partial E_{\theta}}{\partial x} \tag{by defn}\\
\frac{\partial f_{\theta}}{\partial \theta} &= -\eta\frac{\partial^2 E_{\theta}}{\partial x \partial \theta} \tag{diff wrt params}\\
\frac{\partial L}{\partial \theta} &= \frac{\partial L}{\partial f_{\theta}} \cdot \frac{\partial f_{\theta}}{\partial \theta} \tag{chain rule}\\
&=  \frac{\partial L}{\partial f_{\theta}} \cdot -\eta\frac{\partial^2 E_{\theta}}{\partial x \partial \theta} \tag{substitution}\\
\theta_{t+1} &= \theta_t - \alpha \frac{\partial L}{\partial \theta} \tag{GD}\\
&= \theta_t + \alpha \eta\frac{\partial L}{\partial f_{\theta}} \cdot \frac{\partial^2 E_{\theta}}{\partial x \partial \theta} \tag{substitution} \\
\end{align}
$$

How are the dynamics of training with this update different from GD?

Linear layers behave differently?


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
x_{t+1} &\sim \frac{e^{-E(x)}}{\sum_j e^{-E(x_i)}}
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
