Have some uncertainty about the transition fn and the reward fn.
These might be empirical distributions based on data observed so far.

$$
p = P(\tau[s, a, s'], s, a, s'| D) \\
p = P(r[s, a], s, a| D) \\
$$

For a given $\tau, r$, we can solve for $Aut(\tau, r)$ using

How does $Aut(\tau, r)$ change with changes in $\tau, r$?

(ahh. will need approximate symmetries first? actual symmetries will be too sensitive to changes in $\tau, r$)


***

How to find all subgroups of a group.
If we can do so efficiently, then we want to order them by 'complexity'!?
Want to infer from just a couple of observations, not access to P / r.

***

Nodes = states, edges = m(W_{Ps}(s, s'), W_{Rs}(s, s')).
What about the actions!?!? Not clear to me how this graph encodes the similarity between actions, the $g_s(a) = a'$.

Should be Nodes = state-actions?
