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

## Inependent action spaces (disentangled)

Want a set of actions that are orthogonal / modular.

$$
\forall i\neq j, \langle \frac{\partial \tau(s, a_i)}{\partial a_i} , \frac{\partial \tau(s, a_j)}{\partial a_j}  \rangle = 0 \\
$$

Intuition.
- We want the minimal set of action dimensions that span the action space.
- We want actions that act independently (so we are not wasting representational space?).
- ???


E.g. Forward-backward, and left-right action dimensions span the 2D space of actions. There is no need for others. This also allows to __compose__ the actions!

__QUESTION:__ How can we learn a type system!?

## Regularity action spaces

Learn a consistent action space

Want to learn a space where the actions act regularly.

What do we mean by regularly?
- constant
- periodic?
- ???

$$
\begin{align}
\Delta_a \tau(s, a) &= \tau(s, a) - s\\
&= \nabla_a D(\tau(s,a), s)\\
\forall s \;\; : \Delta_a \tau(s, a) &= c \\
\mathop{\text{argmin}}_c \parallel\Delta_a \tau(s, a) - c \parallel_2^2 \\
\mathop{\text{argmin}}_c \sum_i \parallel\Delta_{\omega} \tau(s, \omega_i) - c \parallel_2^2 \;\;: \forall s \\
\end{align}
$$
