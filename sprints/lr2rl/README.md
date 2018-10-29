Meta-RL [@Wang2017LearningTR] trains a learner on the aggregated return over many episodes (a larger time scale). If we construct a temporal decomposition (moving averages at different time-scales) of rewards and aproximate them with a set of value functions, does this produce a heirarchy of meta learners?


### Heirarhcal RL

__Conjecture__: meta/transfer/continual/... learning naturally emerge from heirarchical/multi-scale RL.
We naturally see a decomposition of the tasks into what they share at different levels of abstraction (not just time!?).

__TODO__: Heirarchical filters and meta RL and options.

$$
\begin{align*}
z_t &= f(s_t, a_t, r_t^k) \\
\pi(s_t) &= g(\sum_k f_k(z_t)) \\
v_t^k &= h_k(z_t) \\
\mathcal L_k &= \sum \parallel v_t^k - R(\gamma_k) \parallel \\
R(\gamma) &= \sum \gamma^i r_i \\
\end{align*}
$$

Ahh. But need to be heirarchically propagating forward differences __!!!__ ? Else would need very large compute to fit in large enough batches...

### Resources

- [Learning to learn](https://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/)
