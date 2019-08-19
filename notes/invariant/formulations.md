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
