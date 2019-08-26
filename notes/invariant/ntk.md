## Neural tangent kernel

$$
\begin{align}
\theta_{t+1} &= \theta_t - \eta \nabla_{\theta} \mathcal L(\theta_t) \tag{1}\\
\theta_{t+1} &= \theta_t - \eta \mathop{\mathbb E}_{\tilde x\sim X} \big[\nabla_{\theta}  f(x, \theta_t) \nabla_{f} \ell(x, \theta_t)\big] \tag{2}\\
f(x, \theta') &= f(x, \theta) + \nabla_{\theta}f(x, \theta) (\theta' - \theta) + \mathcal O(\dots) \tag{3}\\
\theta_{t+1} &= \theta', \theta_t = \theta \tag{4}\\
f(x, \theta_{t+1}) - f(x, \theta_t) &= \nabla_{\theta}f(x, \theta) \bigg(- \eta \mathop{\mathbb E}_{\tilde x\sim X} \big[\nabla_{\theta} f(\tilde x, \theta_t) \nabla_{f} \ell(\tilde x, \theta_t)\big]\bigg) + \mathcal O(\dots) \tag{5}\\
\Delta(t) f(x) &= - \eta\nabla_{\theta}f(x, \theta_t) \bigg( \mathop{\mathbb E}_{\tilde x\sim X} \big[\nabla_{\theta} f(\tilde x, \theta_t) \nabla_{f}  \ell(\tilde x, \theta_t)\big]\bigg) + \mathcal O(\dots) \tag{6}\\
\Delta(t) f(x) &= - \eta  \mathop{\mathbb E}_{\tilde x\sim X} \big[\nabla_{\theta}f(x, \theta_t) \nabla_{\theta} f(\tilde x, \theta_t) \nabla_{f}  \ell(\tilde x, \theta_t)\big] + \mathcal O(\dots) \tag{7}\\
\Delta(t) f(x) &= - \eta  \frac{1}{N}\sum_{i=1}^N \nabla_{\theta}f(x, \theta_t) \nabla_{\theta} f(x_i, \theta_t) \nabla_{f}  \ell(x_i, \theta_t) + \mathcal O(\dots) \tag{8}\\
k(x_i, x_j) &:= \langle \nabla_{\theta}f(x_i, \theta_t), \nabla_{\theta} f(x_j, \theta_t) \rangle \tag{9}\\
\Delta(t) f(x) &= - \eta  \frac{1}{N}\sum_{i=1}^N k(x, x_i) \nabla_{f}  \ell(x_i, \theta_t) + \mathcal O(\dots) \tag{10}\\
\end{align}
$$


1. The definition of gradient descent.
1. ???. Chain rule to expand the derivatives.
1. A first order taylor expansion of $f$.
1. Associate the parameter in (2) and (3).
1. Substitute euqation (2) into the $(\theta' - \theta)$ in equation (3),
1. Rewrite $f_{t+1}-f_t$ on LHS as a difference. Also move the learning rate out of the expectation.
1. Move the outer gradient term into the expectation. We can do this because if doesnt depend on $\tilde x$.
1. We estimate the expectation with a (data) set of samples.
1. Define the neural tangent kernel (NTK).
1. Rewrite (8) using the NTK.

__But__. These equations are still dependent on $\theta_t$? I don't understand!?! Oh. I get it. Assuming $\dot\theta = \eta \nabla\mathcal L$ then $\dot y = -\eta H \nabla \ell$.

### Second order??!?

- how big is it?! can we bound it?
- what does the second order term look like!?

$$
\begin{align}
\theta_{t+1} &= \theta_t - \eta \nabla_{\theta} \mathcal L(\theta_t) \\
\theta_{t+1} &= \theta_t - \eta \mathop{\mathbb E}_{\tilde x\sim X} \big[\nabla_{\theta}  f(x, \theta_t) \nabla_{f} \ell(x, \theta_t)\big] \\
f(x, \theta') &= f(x, \theta) + \nabla_{\theta}f(x, \theta) (\theta' - \theta) + \frac{1}{2}\nabla^2_{\theta}f(x, \theta) (\theta' - \theta)^2 + \mathcal O(\dots) \\
\theta_{t+1} &= \theta', \theta_t = \theta \\
\frac{1}{2}\nabla^2_{\theta}f(x, \theta) (\theta' - \theta)^2 &= \frac{1}{2}\nabla^2_{\theta}f(x, \theta)\bigg(- \eta \mathop{\mathbb E}_{\tilde x\sim X} \big[\nabla_{\theta} f(\tilde x, \theta_t) \nabla_{f} \ell(\tilde x, \theta_t)\big]\bigg)^2 \\
&= \frac{-\eta^2}{2} \nabla^2_{\theta}f(x, \theta) \mathop{\mathbb E}_{\tilde x\sim X} \big[\nabla_{\theta} f(\tilde x, \theta_t) \nabla_{f} \ell(\tilde x, \theta_t)\big]^2 \\
\end{align}
$$

## Score fns

$$
\begin{align}
\nabla_{\theta} \log p(x|\theta) &= \frac{\nabla_{\theta} p(x|\theta)}{p(x|\theta)} \\
\theta_{t+1} &= \theta_t - \eta \nabla_{\theta} \mathcal L(\theta_t) \tag{1}\\
\theta_{t+1} &= \theta_t - \eta \mathop{\mathbb E}_{\tilde x\sim p(\cdot | \theta_t)} \big[\nabla_{\theta}  \log p(\tilde x | \theta_t) \ell(x)\big] \tag{2}\\
p(x| \theta') &= p(x| \theta) + \nabla_{\theta}p(x| \theta) (\theta' - \theta) + \mathcal O(\dots) \tag{3}\\
p(x| \theta') &= p(x| \theta) + p(x| \theta) \nabla_{\theta} \log p(x|\theta)(\theta' - \theta) + \mathcal O(\dots) \tag{4}\\
\theta_{t+1} &= \theta', \theta_t = \theta \tag{5}\\
\Delta p(x, t) &= p(x| \theta_t) \nabla_{\theta} \log p(x|\theta_t) \Big(- \eta \mathop{\mathbb E}_{\tilde x\sim p(\cdot | \theta_t)} \Big[\nabla_{\theta}  \log p(\tilde x | \theta_t) \ell(x)\big] \Big) \\
&= - \eta p(x| \theta_t) \mathop{\mathbb E}_{\tilde x\sim p(\cdot | \theta_t)} \Big[ \nabla_{\theta} \log p(x|\theta_t) \nabla_{\theta} \log p(\tilde x | \theta_t) \ell(x)\big]  \\
k(x_i, x_j) &= \nabla_{\theta} \log p(x_i|\theta_t) \nabla_{\theta} \log p(x_j | \theta_t) \\
\end{align}
$$

1.
1.
1. This isnt true?! Taylor expansion for distributions!?


## Representation


$$
\begin{align}
\theta_{t+1} &= \theta_t - \eta \mathop{\mathbb E}_{\tilde x\sim X} \big[\nabla_{\theta}  f(g(x, \phi_t), \theta_t) \nabla_{f} \ell(x, \theta_t, \phi_t)\big] \tag{1}\\
\phi_{t+1} &= \phi_t - \eta \mathop{\mathbb E}_{\tilde x\sim X} \big[\nabla_{\phi}  g(x, \phi_t) \nabla_{g}  f(g(x, \phi_t), \theta_t) \nabla_{f} \ell(x, \theta_t, \phi_t)\big] \tag{2}\\
g(x, \phi') &\approx g(x, \phi) + \nabla_{\phi}g(x, \phi) (\phi' - \phi)  \tag{4}\\
f(g(x, \phi), \theta') &\approx f(g(x, \phi), \theta) + \nabla_{\theta}f(g(x, \phi), \theta) (\theta' - \theta)  \tag{3}\\
f(g(x, \phi'), \theta) &\approx f(g(x, \phi), \theta) + \nabla_{\phi}f(g(x, \phi), \theta) (\phi' - \phi)  \tag{3}\\
f(g(x, \phi'), \theta) &\approx f(g(x, \phi), \theta) + \nabla_{\phi}  g(x, \phi_t) \nabla_{g}  f(g(x, \phi_t), \theta_t)(\phi' - \phi)  \tag{3}\\
\Delta f_{\phi}(x, t) &\approx -\eta \\
k(x_i, x_j) &= \langle \nabla_{\phi}  g(x, \phi_t) \nabla_{g}  f(g(x, \phi_t), \theta_t), \nabla_{\phi}  g(x, \phi_t) \nabla_{g}  f(g(x, \phi_t), \theta_t) \rangle
\end{align}
$$
