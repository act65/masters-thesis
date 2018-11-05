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

### IRL

$$
\begin{align}

\end{align}
$$


### Stretch

- Ok, it has limitations. Not able to oscillate (if we only have a single fn). What about an ensemble? How powerful is that? Or local minimisation of an energy fn shared over space/time?
- If we used this model as part of an ensemble, would it learn to model agents, independently capturing their value functions.
- Could extend to adding more structure into the energy function or update? (e.g. scale by hessian)
- computational complexity versus SGD on NN.
- implement on atari.
- problem. for most atari games, a single frame is not enough to fully specify the state, need the velocity as well.
- is it possible to get non-linear behaviour from many locally convex energy fns? (yes?)
- what if we dont get to observe optimal trajectories? only approximately optimal??? humans often dont know the best way...
- model emergent colloidial structures and attempt to learn the (local) energy function they are optimising.
- will NNs work well in this local setting? quite often they will see nothing/be in some local minima!?
- oh... how is this related to equilibrium propagation?!?
- What problems are ensembles of local losses bad at?


```python
fmap = cnn(x)
energies = tf.map_fn(energy_fn, fmap)
x_t = x - lr*tf.gradient(energies, x)
```

## Resources

- [IRL](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf)
- [Max entropy IRL](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf)
- [GANs, IRL, EBMs](https://arxiv.org/abs/1611.03852)
- [DAC for IRL](https://arxiv.org/abs/1809.02925)
- [IOC](https://arxiv.org/abs/1805.08395)
