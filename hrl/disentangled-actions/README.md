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
