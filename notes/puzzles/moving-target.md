[Diagnosing Bottlenecks in Deep Q-learning Algorithms](https://arxiv.org/abs/1902.10250)

What is the effect of distribution shift and a moving target?

The standard formulation of Q-learning prescribes an update rule, with no corresponding objective function (Sutton et al., 2009a).


To study the distribution shift problem, we exactly compute
the amount of distribution shift between iterations in totalvariation distance, $DT V (µt+1||µt))$ and the “loss shift”:

$$
\mathop{\mathbb E}_{µ_{t+1}} [(Q_t - TQ_t)^2] - \mathop{\mathbb E}_{µ_t} [(Q_t - TQ_t)^2]
$$



> Overall, our experiments indicate that nonstationarities in both distributions and target values, when isolated, do not cause significant stability issues. Instead, other factors such as sampling error and function approximation appear to have more significant effects on performance.
