##### Incremental

(model free)

$z$-iterations. But we need to find a way to project $(s_t, a_t, r_t, s_{t+1}) \to (s_t, u_t, q_t, s_{t+1})$.

\begin{aligned}
z_{t+1}(s_i) = z_{t}(s_i) -\eta \bigg( e^{q(s_i)}z(s{'}_i)^{\gamma} -z_t(s_i)\bigg) \\
z_{t+1}(s_i) = (1- \eta)z_{t}(s_i) + \eta\tilde z(s_i)\\
\tilde z(s) = e^{q(s)}z(s{'})^{\gamma} \\
\end{aligned}

TODO implement!


- Is there a way to learn $p, q$ incrementally!?!?
- What is the essence of what is being done here?

\begin{aligned}
r(s, a) = q(s) - \sum_{s'}P(s' | s, a) \log(p(s'|s)) \\
u^{* }(s'| s) = \frac{p(s' | s)\cdot z(s')^{\gamma}}{\sum_{s'} p(s' | s) z(s')^{\gamma}}  \\
\end{aligned}

\begin{aligned}
q_{t+1}(s) = (1-\eta)\;q_{t+1}(s) + \eta \;r(s_{t-1}, a_{t-1}) \\
p_{t+1}(s'|s) = (1-\eta)\;q_{t+1}(s) + \eta \;r(s_{t-1}, a_{t-1}) \\
u_t = \\
\end{aligned}

So we need counts?!?! The $p(s' | s)$? Hmm. That will be expensive. Or inaccurate. Maybe both.

Want the story to be;
Todorov had some ideas here. But there are fundamental problems that were avoided / overlooked.
That approach only works in a special case.
