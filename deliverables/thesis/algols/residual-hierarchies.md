

## Heirarchical

- [Hierarchy through Composition with Linearly Solvable Markov Decision Processes](https://arxiv.org/abs/1612.02757)
- [Alex Nichol's preirarchy](https://blog.aqnichol.com/2019/07/24/competing-in-the-obstacle-tower-challenge/)


## Residual

- [Residual Reinforcement Learning for Robot Control](https://arxiv.org/pdf/1812.03201.pdf)
- [Residual Policy Learning](https://arxiv.org/abs/1812.06298)
- [Deep Residual Reinforcement Learning](https://arxiv.org/abs/1905.01072)



and boosting?




## Multi-task LMDPs

Work to be done!?

- Extenstion to temporal abstraction?!
- Characterise its complexity
- Guarantees on when a new task is likely to be acculately solved with the current basis.
- Generalise to continuous states / actions
- a 'deep' version of this!? has not been show to scale to more complicated problems.
- under statistical uncertainty!? we might not know $P, r$...


$$
\mathop{\text{argmin}}_{w} \parallel q- Qw\parallel \text{subject to} \;\;Qw \ge 0 \\
$$

Complexity of solving a new task, that is within the span of the existing tasks.
Solve $w = Q^{\dagger}q$. Dont need to do any value iteration, or solve an optimal control problem.

Extenstion to temporal abstraction?!

We require that some $Q_i$ are used less frequently than others.

- (does this imply temporal abstraction!? what does?!)
- (what about a decomposition of $Q$ via SVD!?)


> We augment the state space $S^l = S^l \cup S^l_t$ with a set of $N_t$ terminal boundary states $S^l_t$ that we call subtask states. Semantically, entering one of these states will correspond to a decision by the layer $l$ MLMDP to access the next level of the hierarchy.


> The transitions to subtask states are governed by a new $N^l_t$-by-$N^l_i$ passive dynamics matrix $P^l_t$, which is chosen by the designer to encode the structure of the domain. BAD. How do we choose this!?
