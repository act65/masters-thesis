While learning a model $s_{t+1} = \tau(s_t, a_t)$ is useful. It is more useful to know how to get around using that model. For example, I want to get to $s^k$, how can I do that considering I am in another state, $s^i$? Want a function $f(s^i, s^k) \to \{a1, a_2, \dots, a_n\}$ that outputs a sequence of actions. You need to know how to get around...

Resources

- [HER](https://arxiv.org/abs/1707.01495)
- [Reachability](https://arxiv.org/abs/1810.02274)
