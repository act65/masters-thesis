## Local energies

I am espeically curious about how global energies can be composed of many local energies and how this relates to emergent phenomena. Where the dimensions that the local energies are applied are symmetries of the system and the local energies represent fundamental rules.

$$
\begin{align}
E(s) &= \sum_{i \dots k \subseteq I} E(s_{i \dots k}) \\
\end{align}
$$

Let $s$ be a n dimensional tensor $\in R^n$. Then the global energy can be written as the sum over many dimensions (such as space or time) of the local energies.

All possible states can be described by $s\in S \subset R^n$. The env is described by a set of all possible states, $S$. The current state is an element of that set. For example, a tensor of real values (representing, say, a map).

__But.__
- __Q:__ How do we construct $s$ in the first place to have locally structured dimensions?
- __Q:__ How do we know which dimensions to share the loss over?

What if $x$ was a graph instead? A dimension would be an ordering of the nodes (ordered by reachability or proximity in time).
How do you learn that the x and y dimensions???

- In boids the dimension we are using is for each 'bird'.
- In colloids the dimension we are using is each 'colloid'.
- In classical physics the dimension we are using is time (???). https://en.wikipedia.org/wiki/Principle_of_least_action
- https://en.wikipedia.org/wiki/Gauss%27s_principle_of_least_constraint
- Spin glasses? Phase transitions? Tensor networks and renormalisation?

Clustering the nodes in the graph by similarity (within subset of their embeddings) could find these dimensions!? (at least the boids/colloids?)

#### Other thoughts

- A single scalar energy function has some limitations, it is not able to oscillate. How do local energies add representational power?
- Is it possible to get non-linear behaviour from many locally convex energy fns? (yes?)
- If we used many IEL/IRL modules, would they learn to model the agents acting in say PACman, independently capturing each value function?

#### Resources

- https://arxiv.org/pdf/1801.02124.pdf
- https://arxiv.org/pdf/1403.6508.pdf
