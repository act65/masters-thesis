Why would we want more depth? (greater length of time deps?? which is related to state size!?)

Equivalence between subgoals and options.

$$
g \in f(\mathcal S) \\
\omega \in g(\mathcal A^k) \\
$$

What is the relationship between these two spaces?

A good set of options should "cover" the important parts of state space. That is, all reachable states, using the given options, should cover the subgoals.

$\mathcal R = \{s': \exists \omega_{0:t}, s' =\tau(s, \omega_{0:t})\}: \;\;f(\mathcal S)\subseteq \mathcal R$.



## Options



## Subgoals


Define subgoal.

Need a metric/reward function so we can measure our progress towards the goal. For each subgoal, we need a function such that $f: \mathcal S \to \mathbb R^+$.
Alternatively we could write this as;
- $f: \mathcal S \times \mathcal G \to \mathbb R^+$.
- a representation and a distance metric!? $\langle f(s), g\rangle$
- ???

Should this function be;
- symmetric, injective, surjective, ...?
- differentiable? convex? ...?

### Pros/cons?

Subgoals

- Needs access to a metric or a model
- Must be solved (via MPC or a learned policy)
-

Options

- Easy and cheap to execute (just look up from memory and go)
- Discovery is !?!
- Is rigidly fixed. Cannot change lower level options else learning must start over!?
- ?


Want:
- system that starts with subgoals. But if the same things is computed/optimised many times, turn it into an option. (rate based encodings!?)
- variable sized options!?


### Refs


##### Subgoals

- [Identifying Useful Subgoals in Reinforcement Learning by Local Graph Partitioning](http://www-anw.cs.umass.edu/pubs/2005/simsek_wb_ICML05.pdf)
- [Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation](https://arxiv.org/pdf/1604.06057.pdf)
- [Feudal](https://arxiv.org/abs/1703.01161)
- [HIRO](https://arxiv.org/abs/1805.08296)
- [Visual Reinforcement Learning with Imagined Goals](https://drive.google.com/file/d/1_LJa86wE3bO_j2w7lsN9gvmLJYFKDyIE/view)
- [Exploring Hierarchy-Aware Inverse Reinforcement Learning](https://drive.google.com/file/d/1pw2JOX14iaRzWTz6wAhwoplrvVhmPX7b/view)
- [Automatic Goal Generation for Reinforcement Learning Agents](http://proceedings.mlr.press/v80/florensa18a/florensa18a.pdf)
