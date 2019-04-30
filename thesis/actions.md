General problem.

Want to learn to in settings with;
- ~10,000,000 actions
- unreliable / changing action spaces

## Intuition

The fable of the caterpillar :).

### Interfaces.

Define..

$$
\phi: A \to G \\
$$


Intuition: steering wheel + pedals, keyboard, console, joystick, ...
Relationship to re-learning to use limbs and prosthetics.

## Settings.

Meta learning

Learning to learn the;
- the value of an action
- how to use a new (or perturbed) interface
- ?




## Properties of action abstractions

Want a nice basis for planning and learning.

Properties
- Independence
- Regularity
- Abstract
- Smoothness
- Simplicity
- Composability
- Expressivity/Span/Cover
- Safety
- Efficient compilation
- Modularity (parts can be compiled independently)
- Debugging (traces => credit assignment!?)

How can these emerge out of a simpler constraint?

(also how is this related to type systems!?)

__Most importantly__. If one of these properties is satisfied, what does that tell us about the accuracy and efficiency of learning and/or planning???

##### Independent

Want a set of actions that are orthogonal / modular. (disentangled)

$$
\begin{align}
\Delta s(a) = \mathbb E[\tau(s, a) - s] \\
\\
\forall i\neq j:i,j \in [1:K] \;\; \\
\langle\Delta s(f(w_i)), \Delta s(f(w_j)) \rangle = 0 \\
\end{align}
$$

##### Regular

Learn a consistent action space
Want to learn a space where the actions act regularly.

$$
\begin{align}
\Delta s(a) = \mathbb E[\tau(s, a) - s] \\
\\
\exists c_i\in \mathbb R^d \;\;\text{s.t.} \\
\parallel \Delta s(a_i) - c_i \parallel_2^2 =0\\
\end{align}
$$

Should point in the same direction, and have the same magnitude.
Does it really need to have the same magnitude??
