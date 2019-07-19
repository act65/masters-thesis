Given an MDP and two 'interfaces'. How hard is it to learn in an MDP augmented with the first interface and to then generalise to the other interface?

$$
\begin{align}
M = \{S, A, \tau, r\} \tag{MDP}\\
f: A \to X \tag{interface}\\
f_1^{-1}(x) = a
\end{align}
$$


We must learn given,

$$
\begin{align}
M_{I} = \{S, X, \tau_f, r_f\} \tag{interface augmented MDP}\\
\tau_{f}(s'|s, x) = \tau(s'|s,f^{-1}(x))\\
r_{f}(s, x) = r(s, f^{-1}(x))\\
\end{align}
$$

My question is;
- how much does it cost to learn $M_{I_2}$ given we had a chance to learn $M_{I_1}$?
- how much better can we do if we get to learn ...

how does this scale with many interfaces?

Once the MDP is recovered (if that is possible...), we only need to learn the interface, $f$. So worst case, given a disentangled representation, is ...?
