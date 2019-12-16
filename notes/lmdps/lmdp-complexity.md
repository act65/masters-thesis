Solving an MDP using LMDPs requires three steps.

1. Find a linear embedding of the MDP.
    - $p, q$ s.t. $r(s, a) = q(s) - \text{KL}\Big(P(\cdot | s, a)\parallel p(\cdot | s)\Big)$.
2. Solve the LMDP.
    - $z = QPz^{\alpha}$, $u^{* }(s' | s) = \frac{p(s'|s)z(s')}{\sum_{ \tilde s} p(\tilde s|s)v(\tilde s)}$.
3. Decode the optimal LMDP control.
    - $\pi(a|s) = \mathop{\text{argmin}}_{\pi} \text{KL}\Big(P_{\pi}(\cdot | s)\parallel p(\cdot | s))\Big), \;\; P_{\pi} = \sum_a P(\cdot | s, a) \pi(a | s)$

__Question:__ What is the complexity of each part? How does this compare to a traditional MDP?


## Linear embedding

```python
for state in states:

```

$\mathcal O (|S| \times (|S||A|/2)^{2.3}  )$


## Solve the LMDP

$\mathcal O ((|S|^{2.3}  )$

## Decoding

$\mathcal O( (|S||A|/2)^{2.3} )$


$$
\begin{align}
P_{\pi}(\cdot | s) &= \sum_a P_k(\cdot | s, a) \pi(a | s) \\
\pi &= \mathop{\text{argmin}}_{\pi} \text{KL}\Big(u(\cdot | s))\parallel P_{\pi}(\cdot | s)\Big) \\
0 &=\text{KL}\Big(u(\cdot | s))\parallel P_{\pi}(\cdot | s)\Big) \\
 &= -\sum_{s'} u(s' | s) \log\frac{P_{\pi}(s' | s)}{u(s' | s)}  \\
 \begin{cases}
    x(n), & \text{for } 0 \leq n \leq 1 \\
    x(n - 1), & \text{for } 0 \leq n \leq 1 \\
    x(n - 1), & \text{for } 0 \leq n \leq 1
  \end{cases}
\end{align}
$$


### moved from thesis.

\subsubsection{The complexity of solutions via LMDPs}

\begin{quote}
Is my path actually shorter?
\end{quote}

The whole point of this abstraction was to make the problem easier to
solve. So has it actually made it any easier?

The complexity of solving our abstraction can be broken down into the
three steps;

\begin{itemize}
\tightlist
\item
  linearisation: \(|S| \times \text{min}(|S|,|A|)^{2.3}\)
\item
  solve the LMDP: \(\text{min}(|S|,|A|)^{2.3}\)
\item
  project back: \(???\)
\end{itemize}

Given that the first step was not passed, optimality. We did not continue this characterisation.

\subsection{Scaling to more complex problems}

The next step of developing any RL algorithm would be to generalise it to more 'real' settings.
The real world isn't as nice as the setting we have been working in. There are a few added complexities;

\begin{itemize}
\tightlist
\item
  sample based / incremental
\item
  large / cts state spaces
\item
  sparse rewards
\end{itemize}

So now that we have explored LMDPs, how can we extract their nice
properties into an architecture that might scale to more complex
problems: larger state spaces and action spaces, sparse rewards,
\ldots{}?

\subsubsection{Incremental implementation}

Generalise to a more complex problem. We are only given samples. A first
step to tackling more complex problems.


\subsubsection{Model based}

Learn \(p, q\) based on samples.

$$
\begin{align}
\mathcal L(\theta, \phi) = \mathop{\mathbb E}_{s, a,} \bigg[ r(s, a) - q_\theta(s) + \text{KL}(p_\phi(\cdot | s) \parallel P(\cdot | s, a)) \bigg]\\
\mathcal L(\theta, \phi) = \mathop{\mathbb E}_{s, r, s'} \bigg[r - q_\theta(s) - p_\phi(s' | s) \log \frac{1}{ p_\phi(s' | s)} \bigg]
\end{align}
$$
