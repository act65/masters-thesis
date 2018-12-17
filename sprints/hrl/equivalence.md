The equivalence of options and subgoals.

## Subgoals

Let the reward be the combination of the intrisic and extrinsic goals.


$Q(s, a)$ is the expected discounted reward of choosing the subgoal, $g$ and attempting to achieve it for $k$ timesteps.

= Value of subgoal + value of following subgoal policy in the future.
$$
\begin{align}
Q_{\Omega}(s, g) &= r(s_t) + \gamma  \mathop{E}_{s' \sim \tau(s, \pi_{\omega}(s, g))}[U(s')] \tag{manager}\\
U(s, g) &=  (1-\beta(s, g))E_{s'\sim \pi_{\omega}(s, g)}[r(s')] + \beta(s, g) V^{\pi_{\Omega}}_{\Omega}(s)\\
r_{\omega}(s_t, g) &= Q_{\Omega}(s_{t+1}, g) - Q_{\Omega}(s_t, g) \tag{manager rewards the worker}\\
Q_{\omega}(s_t, g_t, a_t) &= r_{\omega}(s_t, g) + \gamma E_{a \sim \pi(s')}[Q_{\omega}(s_t, g_t, a_t)] \tag{worker}\\
\end{align}
$$

This is a non-standard definition. $r_{\omega}(s_t, g) = Q_{\Omega}(s_{t+1}, g) - Q_{\Omega}(s_t, g)$ But can we justify it? (_intuitively it makes sense that we would reward the worker if it increases the expected rewards!?_)



## Options


$$
\begin{align}
Q^{\pi}_{{\Omega}}(s, w) &= E_{a\sim \pi(s)}[Q^{\pi}_{\omega}(s, w, a)] \tag{over multiple time steps!??!}\\
Q^{\pi}_{\omega}(s, w, a) &= r(s_t) + \gamma  E_{s' \sim \tau(s, \pi(s))}[U(s', w)] \\
U(s, w) &= (1-\beta(s))Q_{\Omega}(s, w) + \beta(s) V_{\Omega}(s)
\end{align}
$$



## Equivalence

> Roughly. Option-critics take the integral of $Q_{\omega}$ to construct $Q_{\Omega}$ while feudal networks take the derivative of $Q_{\Omega}$ to construct $Q_{\omega}$.

$$
\begin{align}
r_{\omega}(s_t, g) &= Q_{\Omega}(s_{t+1}, g) - Q_{\Omega}(s_t, g) \\
&= r(s_t) + \gamma  E_{s' \sim \tau(s, \pi_{\omega}(s)), g \sim \pi_{\Omega}(s)}[U(s', g)] - (r(s_t) + \gamma  E_{s' \sim \tau(s, \pi_{\omega}(s)), g \sim \pi_{\Omega}(s)}[U(s', g)])\\
&= r(s_t) \tag{at convergence}\\
U(s, g) &=  (1-\beta(s))Q_{\omega}(s, \pi_{\omega}(s, g)) + \beta(s) V^{\pi_{\Omega}}_{\Omega}(s) \tag{rewrite} \\
\end{align}
$$

***

- Integration vs differentiation!?
- Is there a nice way to visualise this!? Pictures!?
