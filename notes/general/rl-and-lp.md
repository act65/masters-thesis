It turns out the problem of finding an optimal control within a MDP, can be solved via linear programming. At first, this seems surprising, other optimisation strategies, for example policy iteration and value iteration, are not linear with respect to the policy. So how can linear programming be applied?

Putterman 19?? showed that ...

$$
\mathop{\text{min}}_x c \cdot x \\
Ax \ge b \\
x \ge 0 \\
$$

To understand this, we can imagine ... the value polytope.

Interestingly, policy iteration, can be shown to be equivalent to This makes me wonder, how are value iteration or policy gradient methods for solving MDPs related to interior point methods for solving LPs?

Finally, ... note that our ability to solve sparse LPs has improved vastly over the years (for example ...). This leads me to wonder, how can policy iteration exploit sparse structure of transitions within an MDP to achieve more efficient learning?

Refs

- [Algebraic and topological tools in linear optimisation](http://www.ams.org/journals/notices/201907/rnoti-p1023.pdf)
- [Markov decision problems]()
- [LP and PI]()
- [Value polytope]()
