## Reachability

While learning a model $s_{t+1} = \tau(s_t, a_t)$ is useful. It is more useful to know how to get around using that model. For example, I want to get to $s^k$, how can I do that considering I am in another state, $s^i$? Want a function $f(s^i, s^k) \to \{a_1, a_2, \dots, a_n\}$ that outputs a sequence of actions. You need to know how to get around...

Want the ability to take two states, a, b, or a current state and a goal state. And to generate a policy/trajectory that will take us there.
Pick two random locations. Start, goal, $s_i, s_g$ and generate a trajectory/policy.

$$
(a_0, a_1, \dots, a_n) \sim \xi(s_i, s_g)  \\
a_i = \pi (s_t, s_g) \\
$$


Potential solutions?

- MPC?
- A heirarchy of transition fns
- ?

The problem is that

Resources

- [HER](https://arxiv.org/abs/1707.01495)
- [Reachability](https://arxiv.org/abs/1810.02274)
- [Topographic memory](https://arxiv.org/abs/1803.00653)
