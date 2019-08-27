## Value symmetries

### Q-learning

Preconditioner.

$$
\begin{align}
\chi \in R^{|S| |A| \times |S| |A|} \\
Q_{t+1} = Q_t - \eta  \frac{\partial \mathcal L}{\partial Q} \cdot \mathcal X \cdot \frac{\partial Q}{\partial \theta}\tag{preconditioner}\\
\end{align}
$$

Sampling distribution.

$$
\chi: |S||A| \times |S||A| \to [0, 1]\\
\mathcal L_{sym} = \mathop{E}_{(s, a), (s', a')\sim\chi} \parallel Q(s, a) - Q(s', a') \parallel_2^2 \\
\mathcal L_{MSE-TD} =  \parallel Q(s, a) - T(Q)(s, a) \parallel_2^2
$$


These two end up being the same thing?!


### Policy gradients

$$
\nabla J(s) = R(s) \cdot \sum_a \nabla \log \pi(a|s) \\
\nabla J(s) = R(s) \cdot \chi \cdot \sum_a \nabla \log \pi(a|s) \\
$$

Distributes reward to other similar states.

### Actor critic




## Averaging over orbits
Turns out ... the same.

$$
\theta_{t+1} = \theta_t - \eta \nabla \mathcal L(\theta) \\
$$

### Weight sharing

$$
f(x, \theta) = [g(x[1], \theta),\dots ,g(x[k], \theta)] \\
\theta_{t+1} = \theta_t - \eta \frac{1}{N}\sum_{i=1}^N  \frac{1}{K}\sum_{j=1}^K  \nabla \ell(x_i[j], \theta) \\
$$

- so, if all the $x[i]$ s happen to be the orbit of a single $x_i[0]$ then this update is the same as one via data augmentation.
- How does this realte to CNNs!? They are equivariant, not invariant.

### Label sharing

$$
\mathcal L(\theta) = \mathop{\mathbb E}_{x, y, x', y' \sim \chi} \ell(x, y, \theta) + \ell(x, y', \theta) + \ell(x', y, \theta) + \ell(x', y', \theta)\\
$$


### Data augmentation

$$
g \in G \\
\mathcal L(\theta) = \mathop{\mathbb E}_{x \sim p(X)}\mathop{\mathbb E}_{g \sim q(G)}\ell(g\circ x, \theta) \\
\theta_{t+1} = \theta_t - \eta \mathop{\mathbb E}_{x \sim p(X)}\mathop{\mathbb E}_{g \sim q(G)} \nabla\ell(g\circ x, \theta)   \\
$$

### Gradient augmentation

$$
\theta_{t+1} = \theta_t - \eta \mathop{\mathbb E}_{g \sim q(G)} g\circ \big(\mathop{\mathbb E}_{x \sim p(X)}\nabla\ell( x, \theta) \big)  \\
$$

Does this even make sense?!?

### Gradient coupling

$$
\frac{\partial f}{\partial \theta} \cdot \chi \cdot \frac{\partial \ell}{\partial f}\\
\theta_{t+1} = \theta_t - \eta \mathop{\mathbb E}_{x \sim p(X)}\mathop{\mathbb E}_{g \sim q(G)} \nabla\ell(g\circ x, \theta) \\
$$

### Output coupling

$$
\mathcal L(\theta) = \mathop{\mathbb E}_{x \sim p(X)}  \bigg[\ell(f(x, \theta)) + \mathop{\mathbb E}_{g \sim q(G)}\parallel [f(x, \theta)]{_{\text{stop}}} - f(g(x),\theta) \parallel \bigg]\\
\theta_{t+1} = \theta_t - \eta \mathop{\mathbb E}_{x \sim p(X)}\mathop{\mathbb E}_{g \sim q(G)} \nabla\ell(g\circ x, \theta)   \\
$$

- This should def be equivalent to label sharing

***

But. What are the advantages of one or the other?
