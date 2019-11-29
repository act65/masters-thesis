From the data we recieve. Trajectories. How can we infer symmetries in the environment / value / ...

> What is the general formulation of finding symmetries?

Find $f$ s.t. $g(f(x)) = g(x)$.
- $x$ could be the state, an action, a reward, ...
-

***

Find a function $f$ that transforms the state in a way that preserves the change in state of each action.

$$
\mathop{\text{argmin}}_{f} \mathop{E}_{(s, a)}^{(\hat s, \hat a)} \mathbf 1_{a = \hat a}\parallel (\tau(s, a) - s) - (\tau(f(\hat s), a) - f(\hat s)) \parallel_2^2 \\
\mathop{\text{argmin}}_{f} \mathop{E}_{(s, a, s')}^{(\hat s, \hat a, \hat s')}\mathbf 1_{a = \hat a}\parallel (s' - s) - (f(\hat s') - f(\hat s)) \parallel_2^2 \\
$$

Many there are many possible $f$s. Could learn an orthogonal ensemble of them?

***

Find a transformation that

$$
T(Q)(s, a) = r(s, a) +  \gamma \mathop{E}_{s' \sim P(\cdot|s, a)} \mathop{\text{max}}_{a'} Q(s', a') \\
T(Q)(s, a) - Q(s, a) = \delta(s, a) \\
$$

If $\delta(s, a) \approx \delta(s', a')$ what does that tell us!?
What if we had $f, g$ such that $\delta(s, a) \approx \delta(f(s), g(a))$?


***
Want a representation that captures the symmetries / invariances in the environment.

- [A Theoretical Analysis of Contrastive Unsupervised Representation Learning](https://arxiv.org/abs/1902.09229)
- [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)
- [Probabilistic symmetry and invariant neural networks](https://arxiv.org/abs/1901.06082)

Representation learning for RL

- [A Geometric Perspective on Optimal Representations for Reinforcement Learning](https://arxiv.org/abs/1901.11530)
- []()


***
[On Variational Bounds of Mutual Information](https://arxiv.org/abs/1905.06922)
[Mutual information neural estimation](https://arxiv.org/abs/1801.04062)
[On mutual information maximisation for representation learning](https://arxiv.org/abs/1907.13625)
Mutual information doesnt work because?!?
Do some minimal tests to show MI is not what we want!?


## What do we want?
> The ability to identify symmetries in state-actions-rewards/value and use that knowledge to share rewards/values between 'similar' state-actions.

How do we know two state-action-reward/values are similar?

- because we have seen it (Boring.)
- because it follows a pattern we have observed so far (rotations of 45, 90, 135, 180, all are similar, therefore 225, 270, 315 are also - probably - similar) (!!!)

To be able to share, what do we need to know?

- That $x, y$ are similar. I.e. that $g \in G: y = g\circ x$
- Just share between all invariant datapoints?! No. This is about generalisation!!

### Inferring symmetries in data

Pick a simpler setting. No noise. Discrete domain. No action.
First we need to be able to identify group structures from observations of ???.
Then, we can generalise to observations of the groups action on another set.
Then, we can generalise to noisy observations.

***


> Ok, what data do we need to be able to infer a group structure $(Z_2, S_4, A_3, Di_8)$?

- Some vs all elements (if we need all, then isnt really an inference problem...).
<!-- (although, observing all the group elements might not be so bad, when they act on a large set!?) -->
- Pairs $c = a \circ s$ vs triples $c = a \circ ?$ (where do the triples come from?)

What information could be provided, or needs to be inferred?

- Number of elements in the group.
- The $n\times n$ relations.
- The type of group (cyclic, alternating, Sporadic) [ref](https://en.wikipedia.org/wiki/Classification_of_finite_simple_groups)
- The identity of subgroups. $Z_2 \times X = \text{Obs}$

Under which constraints?

1. Identity
2. Inverse
3. Closure
4. Commutative / symmetric

***

- Impossible without triples. TODO prove.
- If we are given triples, then we have the job of matrix completion. That needs to satisfy [1,2,3,4].
- We can form triples from pairs when if we know that the transformations are linear?!
- ?

#### Cayley completion
<!-- Expected to find something on the net about this. matrix completion of cayley tables. Am I thinking about it wrong? -->

Ok. We guessed a $n$ (or it has been given) and now we have an incomplete cayley matrix: we filled it in with some observations.

> __Q:__ What is our earch space of possible cayley matrices? How large is it?
How does a new piece of data reduce the number of possible cayley tables?

### How easy is it to solve this inference problem?

Want to have an inductive bias towards simpler symmetries. But, how can we do this without needing to represent all possible symmetries?


### Generalisation to ???

If we can solve: infer group structure from missing data.
Then we can solve:

### Notes

- What about symmetries that are products of subgroups? $S = Z_2 \times Z_3$?
Are they easier to infer?
- Within the same $n$. Is there a notion of more or less complex group structures??
- Need to show that NNs dont have the right symmetric inductive bias. They dont generalise.

Examples

- Knowing that; range $= [0,360)$, and $0, 45, 90, 135, 180$, all are similar. I quess that we are in cyclic group $8$ and therefore $225, 270, 315$ are also similar. Key is that I know that $0, 45, 90, 135, 180$ are related by $0+0, 0+45, 0+45+45, 0+45+45+45, 0+45+45+45+45$.
- Cart pole. $V^{\pi(s, a)}(s') = V^{\pi(-s, -a)}(-s') \forall s'$.
