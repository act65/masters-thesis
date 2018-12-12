While learning a model $s_{t+1} = \tau(s_t, a_t)$ is useful. It is more useful to know how to get around using that model. For example, I want to get to $s^k$, how can I do that considering I am in another state, $s^i$? Want a function $f(s^i, s^k) \to \{a_1, a_2, \dots, a_n\}$ that outputs a sequence of actions. You need to know how to get around...

Resources

- [HER](https://arxiv.org/abs/1707.01495)
- [Reachability](https://arxiv.org/abs/1810.02274)
- [Topographic memory](https://arxiv.org/abs/1803.00653)


Options -> model free RL. !?
Reachability -> model based RL!? (distill model into a goal conditioned planner)
Unsupervised options -> !?


https://arxiv.org/pdf/1811.07819.pdf !?


Reachability

Pick two random locations. Start, goal, $s_i, s_g$ and generate a trajectory/policy.

$$
(a_0, a_1, \dots, a_n) \sim \xi(s_i, s_g)  \\
a_i = \pi (s_t, s_g) \\
$$

Trained on !? Proximity to $s_g$? (but how can we measure that!?)

How to learn this?
- Possible to use the model to discover/learn options!?!
- Or gather data in the real world.


What about $f(s_i, s_g) \in [0, 1]$ so that if $s_g$ is reachable from $s_i$ then is 1 else 0. TDMs!?



Relationship to knowledge base completion. Or link prediction!?
