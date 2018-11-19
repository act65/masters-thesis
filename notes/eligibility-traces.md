
$$
\begin{align}
Q_{t+1}(s_t, a_t) &= Q_t(s_t, a_t) + \alpha \delta_t e_t(s_t, a_t) \tag{for all $s, a$} \\
\Delta Q &= \alpha \delta_t e_t(s_t, a_t) \\
\end{align}
$$

Problem. Need to update for all state-action pairs. How can this be solved efficiently?
Use a discrete representation... (neural discrete AE or the results of some clustering?)


$$
\begin{align}
e_t &= \begin{cases}
\gamma \lambda e_{t-1} + 1, & s=s_t, a=a_t \\
\gamma \lambda e_{t-1}, & s\neq s_t, a \neq a_t\\
\end{cases} \tag{if binary features, ie tabular} \\
e_t &= \gamma \lambda e_{t-1} + \nabla_{\theta_t} Q_t(s_t, a_t) \tag{generalised to cts}\\
e_t &= \gamma \lambda e_{t-1} + \nabla_{\theta_t} V_t(s_t)
\end{align}
$$

So we are keeping an exponential average of the gradient of the state(-action) value with respect to each parameter.

But how does this work? Do we need a window around a cts value?

PROBLEM. To work the various input states and actions would need to be normalised???

Why are these attractive? Keeps the trace of a trajectory lingering. Could simply use the trajectory, but it needs to be feed in, thus we need to store the obs and action. Rather the trace and be stored internally, (hopefully more cheaply!?).

https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2007-49.pdf

***

Ok, so is there a difference to momentum?
https://distill.pub/2017/momentum/

$$
\begin{align}
m_t &=  \beta m_{t-1} + \nabla L_{\theta} \\
\Delta \theta &= -\eta m_t \\
\end{align}
$$

Hypothesis: Any algorithm using momentum (within the optimisation) is using eligibility traces!?

No, not true. In momentum, the past gradients of the loss are (exponentially) averaged and then used to update the parameters. Eligibility traces, are an exponential moving average of the outputs, not the loss.


In the above formulation, $\delta$ captures information about the current reward prediction error, and $e$ captures information about parameters/actions that are likely to have resulted in the RPE.

Could this all be summarised within momentum? No. Momentum is the past RPE's effecting future updates. We want capture the past parameter settings/actions and their effect on current/future rewards. Eligibility traces have nothing to do with how wrong you were wrong in the past. They are trying to assign credit for the current RPE.

Want $\frac{\partial L}{\partial \theta}(s_t, a_t)$ but we dont know it. Instead we can approximate it with...?!?



***

Is there a relationship to second order information?

$$
\begin{align}
\Delta \theta &= \alpha \frac{\partial L}{\partial \theta} \odot \frac{\partial Q}{\partial \theta} \\
\end{align}
$$

***

Integrated gradients = causal effect = eligibility trace?

https://arxiv.org/abs/1703.01365


***

Resources

- http://incompleteideas.net/book/ebook/node72.html
- http://pierrelucbacon.com/
- http://web.eecs.umich.edu/~baveja/Papers/ICML98_LS.pdf
- https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2007-49.pdf
- https://amreis.github.io/ml/reinf-learn/2017/11/02/reinforcement-learning-eligibility-traces.html
- http://rl.cs.mcgill.ca/tracesgroup/
