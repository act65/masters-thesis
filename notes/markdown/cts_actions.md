If $Q$ is a large non-linear neural network, then solving the equation below can be hard.

$$
\mathop{\text{argmax}}_a Q(s, a)
$$

Solutions?

- Do sgd on $Q(s, a)$.
- Learn $Q(s, a)$ so that Q is convex wrt a?!
