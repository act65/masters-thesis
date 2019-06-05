How to deal with uncertainty?

## Uncertain transitions.

We should only get less certain about future states.

$$
H(s_t) \le H(s_{t+1}) \\
$$

Want to enforce this prior.

- In the tabular case, this naturally occurs as a function of the row normalisation.
- In a function approximation case, how can we enforce this prior?

$$
R(s_t, a_t) = \big | \log(\frac{H(\tau(s_t, a_t))}{H(s_t)} ) \big | \\
$$

Properties.
- Asymmetric. Care a lot if $H(s_t) \le H(s_{t+1})$ but not so worried about $H(s_t) \ge H(s_{t+1})$.
- ?
