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

A, B are similar if there exists an easy to find, structure preserving, mapping from A to B. While two things might be isomorphic, depending on what we know about them it could be very hard to find a mapping from one to the other.

A representation $h = f(x)$ is a good one if, for all tasks, we can easily find similarities between __any__ two tasks.

So a better representation would be easily able to map between more tasks/domains.

## Graph similarity

Generate various graphs that share some known structure.
Want to be able to reason about symmetries in graphs or manifolds at many different scales and locations.

## Shared basis

Could imagine an over-complete basis, shared between task A and B?

Generate a basis, $\mathcal B$, maybe just a set of vectors.
Subsample and use to generate tasks. $\mathcal T_i \subset \mathcal B$.
