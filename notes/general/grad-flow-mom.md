\section{Continuous flow and its discretisation}

A linear step of size, \(\alpha\), in parameter space, ie by gradient
descent, is not necessrily a linear step in parameter space.

\includegraphics[width=0.3\textwidth,height=0.25\textheight]{../../pictures/figures/vi_sgd-vs-vi_mom_0001.png}
\includegraphics[width=0.3\textwidth,height=0.25\textheight]{../../pictures/figures/vi_sgd-vs-vi_mom_001.png}
\includegraphics[width=0.3\textwidth,height=0.25\textheight]{../../pictures/figures/vi_sgd-vs-vi_mom_01.png}

This is consistent with acceleration of gradient descent being a phenomena only possible in the discrete time setting. (see \cite{Betancourt2018} for a recent exploration)

This phenomena can be explained by the exponential decay of the momentum terms.

\begin{align}
m_{t+1} = m_t + \gamma\nabla f(w_t) \\
w_{t+1} = w_t - \eta (1-\gamma) m_{t+1} \\
\end{align}

As \(\eta \to 0\), \((1-\gamma) \cdot m_{t+1} \to \nabla f(w_t)\).

TODO, prove it.
