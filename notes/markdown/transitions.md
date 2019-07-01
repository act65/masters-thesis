Learning a transition function.

Is learning from $(s_t, s_{t+2})$ strictly better than learning from $(s_t, s_{+1})$.
How does this help learn longer dependencies?
It learns to account for its errors?


### Why is next-step prediction not sufficient?

> We argue that next-step prediction exacerbates the conditioning problem described in Section 2.1. In a physically realistic environment the immediate future observation $xt+1$ can be predicted, with high accuracy, as a near-deterministic function of the immediate past observations $x(t−κ),...,t$. This intuition can be expressed as $p(xt+1|x(t−κ),...,t, a(t−κ−1),...,(t−1), b(t−κ)) ≈ p(xt+1|x(t−κ),...,t, a(t−κ−1),...,(t−1))$. That is, the immediate past weakens the dependency on beliefstate vectors, resulting in ∇btL ≈ 0. Predicting the distant future, in contrast, requires knowledge of the global structure of the environment, encouraging the formation of belief-states that contain that information. [Shaping Belief States with Generative Environment Models for RL](https://arxiv.org/abs/1906.09237)
