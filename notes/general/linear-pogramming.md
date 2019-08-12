It turns out the problem of finding an optimal control within a MDP, can be solved via linear programming (LP). This seems surprising as other optimisation strategies for RL, for example policy iteration and value iteration, are non-linear / non-convex optimisation problems. Why are we not exploiting the problem's linearity?

Putterman 19?? showed that an MDP can be solved via this formulation as a LP.

$$
\mathop{\text{min}}_x c \cdot x \\
Ax \ge b \\
x \ge 0 \\
$$

To understand this, we can imagine ... the value polytope.

The simplex method is one of the more successful solvers of LPs. The simplex method works by ... . 
It turns out that the simplex method, with X pivoting rule, is actually equivalent to policy iteration.

Next. Interior point methods are another way to solve LPs. What is their connection to policy gradient methods?

Finally, ... note that our ability to solve sparse LPs has improved vastly over the years (for example ...). This leads me to wonder, how can policy iteration exploit sparse structure of transitions within an MDP to achieve more efficient learning?

Refs

- [Algebraic and topological tools in linear optimisation](http://www.ams.org/journals/notices/201907/rnoti-p1023.pdf)
- [Markov decision problems]()
- [LP and PI]()
- [Value polytope]()
