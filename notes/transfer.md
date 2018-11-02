What is necessary for transfer?

- memory of the past
- a way to project old representations into ...?
- ?

What is sufficient?

- increased performance after learning
- ?


Not necessarily a compression?
How does transfer relate to;

- compression
- minimisation of energy/resources (necessarily leads to transfer!? if there is structure...)
- Generalisation = zero shot learning? (a good guess)
- Transfer = one shot learning (given information about your target task, find similarities to past tasks)

***

Two environments, $t_i, t_j$ are similar if there exists an easy to find, structure preserving, mapping $f$ between them $f(t_i) = t_j$. While two environments might be isomorphic, depending on what we know about them it could be very hard to find a mapping from one to the other (at least polynomial time).

- we need an approximate notion of equality, a distance or metric

$$
\begin{align}
f^* &= \mathop{\text{argmin}}_f d(f(t_i), t_j) \\
C(t_i, t_j) &= ??? \tag{the cost of finding $f^* $}\\
C(t_i, t_j) &\approx d(f^* (t_i), t_j) \tag{fixed computational budget}\\
\end{align}
$$

- What if $\text{min} \;d(f^* (t_i), t_j) >> 0$?
- ?

A representation $g: t \to h$ is a good one if, for many tasks, we can easily find similarities between any two tasks, $f_{ij}(g(t_i)) = g(t_j)$.

So a better representation would be easily able to map between more tasks/domains.

$$
\begin{align}
L &= E_{t_i \sim \mathcal T} E_{t_j \sim \mathcal T} [C(t_i, t_j)]\\
\end{align}
$$

Could apply this notion to the various $f_{ij}^{* }$ as well. How easily can they be transformed into each other?

- how much compute should be spent on trying to find similarities? when it could just be spent on learning the new domain. it depends on memory constraints...

***

Or. A good representation is one that helps in different tasks. We culd assign credit via ???.

## Graph similarity

Generate various graphs that share some known structure.
Want to be able to reason about symmetries in graphs or manifolds at many different scales and locations.

## Shared basis

Could imagine an over-complete basis, shared between task A and B?

Generate a basis, $\mathcal B$, maybe just a set of vectors.
Subsample and use to generate tasks. $\mathcal T_i \subset \mathcal B$.
