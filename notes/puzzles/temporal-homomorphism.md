## The MDP homomorphism
$$
\begin{align}
\mathcal H&: M \to M \\
\tilde P(f(s')|f(s), g_s(a)) &= \sum_{\tilde s'\in {[s']}_f } P(\tilde s'| a, s) \\
\tilde r(f(s), g_s(a)) &= r(s, a) \\
\end{align}
$$

Hmm. Shouldnt it be something more like below?!?

$$
\begin{align}
\sum_{s'\in {[\tilde s']}_f} \sum_{a \in {[\tilde a]}_{g_{\tilde s}}} \sum_{s \in {[\tilde s]}_f} P(s'|s, a) = P(\tilde s'| \tilde a, \tilde s) \\
\end{align}
$$
No. Summing over $s \in {[\tilde s]}_f$ is saying we are in all of these states at once.
Similarly, Summing over $a \in {[\tilde a]}_{g_{\tilde s}}$ is saying we are taking all of these actions at once.
