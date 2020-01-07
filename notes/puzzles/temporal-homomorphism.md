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

## A temporal transformation

(is this actually a homomorphism!? what do I need to prove?)

$$
\begin{align}
\Omega_k&: M \to M \\
\Omega_k&:\{S, A, r, P, \gamma\} \to \{S, A^k, \hat T_k(r), \hat T_k(P), \gamma\} \\
\hat T_k(r) &= \sum_{i=t}^{t+k} \gamma^{i-t} r(s_i, a_i) \\
\hat T_k(P) &= \prod_{i=t}^{t+k} P(s_{i+1} | s_i, a_i) \\
\end{align}
$$

Need to show two temporally transformed MDPs can be composed? This allows us to construct options of various length!? $\Omega_i(M) \wedge \Omega_j(M) \to \tilde M$.

***

What about a goal-like verion of this?


## The MDP temporal homomorphism

http://www.cse.iitm.ac.in/~ravi/papers/IJCAI03.pdf ??? Doesnt seem satisfying!?

$$
\begin{align}
\Omega_k: M \to M \\
\mathcal H: M \to M \\
\Omega_k\circ \mathcal H: M \to M \tag{temporal homomorphism}
\end{align}
$$

Will abstract based on similarities between state options, $(s, \omega)$.

***

What if I wanted to do state-action and state-option abstraction?!


***

But this gets inefficient if we want to do this for high values of $k$.
Is it possible to build up using larger and larger values of $k$.

If we know that $[11] = [00]$ and $[012] = [021]$, then we know that $[01211] = [02100]$?
How does this help us reduce the computational complexity?

So there is an advantage to building your temporal abstraction from k=n first, then to k=n+1. Depends on how many symmetries you ave found?
