
$$
\begin{align}
w^{t+1} &= w_1^{t+1}\cdot w_2^{t+1} \\
&= (w_1^t - \eta (1-\gamma)\sum_{\tau=0}^t \gamma^{t-\tau} \nabla_{w_1^{\tau}}) \cdot (w_2^t - \eta (1-\gamma)\sum_{\tau=0}^t \gamma^{t-\tau} \nabla_{w_2^{\tau}}) \\
&= (w_1^t - \eta m_1^t) \cdot (w_2^t - \eta m_2^t) \\
&= w^t - w_1^t\eta m_2^t - \eta m_1^tw_2^t + \mathcal O(\eta^2) \\
&\approx w^t - w_1^t\eta (1-\gamma)\sum_{\tau=0}^t \gamma^{t-\tau} \nabla_{w_2^{\tau}} - \eta (1-\gamma)\sum_{\tau=0}^t \gamma^{t-\tau} \nabla_{w_1^{\tau}}w_2^t \\
&= w^t
\end{align}
$$
