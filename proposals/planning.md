%planning
# Planning

CFRM, MCTS, ... cool.
What about planning in with continuious actions?  -> LQR, MPC


__Q:__ How much harder is planning in a stochastic system than in a deterministic system?!?

## Model predictive control

(what about LQR, ...)

Short coming of MPC. Finite horizon.
- Will be short sighted/myopic? will fail at problems with small short term reward in wrong direction?
- Cannot trade off width for depth.
- What if the state returned by the model is a distribution? Then we need to explore possibilities!?!?

Can derive Kalman filters from these?!!

$$
\begin{align}
V(x) = min_u [l(x, u) + V(f(x, u))] \tag{the cts bellman eqn!?}\\
\end{align}
$$

> The Q-function is the discrete-time analogue of the Hamiltonian

## Backwards

What are the advantages of having access to a inverse dynamics model?

- Some problems are easier to solve? (U-problem?)
- Smaller search space (1 vs 2 circles?)
- ?

## General problem

Need to integrate a dynamical system.
But how to do this when it is;
- stochastic?
- biased?

Want to learn to integrate!?

## Resources

- [Differential Dynamic Programming](https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf)
- [mpc.torch](https://locuslab.github.io/mpc.pytorch/)
- ?
