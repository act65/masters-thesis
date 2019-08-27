## Intro: Symmetry.

> Want a representation that makes the problem easier to solve.

What is it?
Why do we care?
Invariants, equivariants. Quotients.


- Find the symmetry.
- Solve MDP in only the quotient.
- Map back to original.




### Symmetries and and sample efficiency

Size of the initial space $n = \mid S \mid, m = \mid A \mid$.
Size of the quotient space $\tilde n = \mid S \mid, \tilde m = \mid A \mid$.
(need to derive the relationship between the two depending on the amount of symmetry)

Therefore, if we have inferred the structure of our MDP, then solving it requires $\mathcal O()$ samples, rathern than $\mathcal O(?)$

But what is the sample complexity of learning the symmetry?!?!!
<!-- Mention / prove something about conv nets?! -->


### Invariant representations for RL

So for RL, which measures of similarity make sense?

Use $\chi$ to say that two action-state values are simiar because;
- similar returns
- similar ???
- similar (discounted) state trajectories
- they can reach many of the same states (within k actions)
-

### Learning a measure of symmetry / similarity

<!-- awesome, we can see why symmetry is interesting. and how to use it if we know it. but... how do we find it? -->


Without a measure of 'sameness' / 'similarity' (aka a metric?). Symmetry does make sense.
A symmetry is defined as the conservation of a ? under transformations.
We need a measure of that conserved quantity if we want to

For example; an apple classifier oracle. It tells us that a picture of an apple is still an apply if rotated, translated, sharpened, ...



__Problem__. The policy. How can we disentangle which ever policy we are currently using from the underlying symmetries in the environment?
(is this just asking about learning a model?! $P, r$?)
