Problem.

We have two models of dynamics. A model of human behaviour, and a model of

$$
\begin{align}
s_{t+1} &= \tau_3(\tau_1(s_t), \tau_2(s_t)) \tag{ground truth} \\
s_{t+1} &\approx f(x) + g(x) \tag{ensemble}\\
\end{align}
$$

Is there a case when a simple sum of experts can work? The

$f,g$ will sometimes depend on and alter different subsets of the states. (what I do in my bedroom doesnt effect global politics)
But, other times, two models will both have beliefs about what should happen next. How can we combine these beliefs?

The models might interfere with each other. While they might independently produce plausible states, there is no guarantee that their combination will produce a plausible state.
A projection/constrain problem?! Kinda.

Integration test!

$$
\begin{align}
s_{t+1} = \mathop{\text{argmax}}_{s_{t+1}} \prod_i p_i(s_{t+1} \mid s_t)
\end{align}
$$

problems

- expensive?
- deterministic?
