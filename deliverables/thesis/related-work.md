## Related work

#### MDPs

Dynamic programming, linear programming, ...?

$$
Q^{\pi}(s_0, a_0) = r(s_0, a_0)
+ \gamma \mathop{\text{max}}_{a_1} \mathop{\mathbb E}_{s_1\sim p(\cdot | s_0, a_0)} \Bigg[ r(s_1, a_1)
+ \gamma \mathop{\text{max}}_{a_2} \mathop{\mathbb E}_{s_2\sim p(\cdot | s_1, a_1)} \bigg[r(s_2, a_2)
+ \gamma \mathop{\text{max}}_{a_3} \mathop{\mathbb E}_{s_3\sim p(\cdot | s_2, a_2)} \Big[
\dots \Big] \bigg] \Bigg]
$$

##### HRL

Temoral abstractions of actions.(how does this related to a decomposition of rewards)
Ok, so we wany a multiscale representation?
Understanding how actions combine (this is necessary knowledge for HRL?)


Reasons to do HRL??? (want to verify these claims - and have refs for them)

- credit assignment over long time periods (learning faster in one env)
- exploration
- transfer

- To learn action abstractions they must capture info about the model. How much harder is it to learn action abstractions in model-free vs model-based settings?
- Reward as a function of a subspace of the state space. (this is important for learning abstract representations and actions!?)
- What do cts linear heirarchical actions look like!? and their loss surface!?

- [HLMDPs](https://arxiv.org/abs/1612.02757)
- [Modulated policy heirarchies](https://arxiv.org/abs/1812.00025)
- [Model free representations for HRL](https://arxiv.org/abs/1810.10096)
- [Prierarchy: Implicit Hierarchies](https://blog.aqnichol.com/2019/04/03/prierarchy-implicit-hierarchies/)
- Options
- [Near optimal representation learning for heirarchical RL](https://openreview.net/forum?id=H1emus0qF7)


Relation to pretraining / conditioning?

#### Dynamic programming

What is it? Memoized search.
Why should we care?


## Model-based RL

Pros and cons.


Model-based learning can be bad...
There may be many irrelevant details in the environment that do not need to be modelled.
A model-free learning naturally ignores these things.

The importance of having an accurate model!

For example, let $S\in R^n$ and $A\in [0, 1]^n$. Take a transition function that describes how a state-action pair generates a distribution over next states $\tau: S \times A \to \mathcal D(S)$. The reward might be invariant to many of the dimensions. $r: X \times A -> \mathbb R$, where $X \subset S$.

Thus, a model mased learner can have arbitrarily more to learn, by attempting to learn the transition function. But a model-free learner only focuses on ...

This leads us to ask, how can we build a representation for model-based learning that matches the invariances in the reward function.
(does it follow that the invariances in reward fn are the invariances in the value fn. i dont think so!?)

Take $S \in R^d$ and let $\hat S = S \times N, N \in R^k$. Where $N$ the is sampled noise. How much harder is it to learn $f: S \to S$ versus $\hat f: \hat S \to \hat S$?

https://arxiv.org/pdf/1903.00374v3.pdf
https://arxiv.org/abs/1907.02057

## Representation learning and abstraction

The goal is to find a representation that decomposes knowledge into its parts.

Another way to frame this is: trying to find the basis with the right properties.

- sparsity,
- independence,
- multi scale,
- locality/connectedness
- ???
