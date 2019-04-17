Insert quote(s) from Marta White. About safety.
Also want to flesh out my intuition about how we can learn about something we havent done...

(it isnt really something we havent done!?)
Also, could have come from someone else!? Off-policy learning could be used to learn from others experiences.

***

> In this case, $\pi$ is the target policy, $b$ is the behavior policy, and both policies are considered fixed and given.
In order to use episodes from $b$ to estimate values for $\pi$, we require that every action
taken under $\pi$ is also taken, at least occasionally, under $b$. That is, we require that
$\pi (a|s) > 0$ implies $b(a|s) > 0$. This is called the assumption of coverage.

ok. So 'coverage' means that our exploration policy will take the optimal action with non-zero probability.
But the lower the probability is the harder it is to learn the value of the optimal action...


## Importance sampling

ref - https://statweb.stanford.edu/~owen/mc/Ch-var-is.pdf

> In many applications we want to compute $µ = E(f(X))$ where $f(x)$ is nearly
zero outside a region $A$ for which $P(X ∈ A)$ is small. The set A may have
small volume, or it may be in the tail of the $X$ distribution. A plain Monte
Carlo sample from the distribution of $X$ could fail to have even one point inside
the region $A$.

Aka. Low frequency, high value.

> Importance sampling is more than just a variance reduction method. It can
be used to study one distribution while sampling from another.

$$
\begin{align}
\mu &= E_p(f(x)) \\
&= \int_D p(x)f(x) dx \\
&= \int_D \frac{p(x)f(x)}{q(x)} q(x) dx \\
&= E_q(\frac{p(x)f(x)}{q(x)}) \\
\end{align}
$$

Problem when $q(x) \approx 0$. The importance ratio gets large.

$$
\begin{align}
\rho(x) &= \frac{p(x)}{q(x)} \\
\mu &= \frac{\int \rho(x) f(x) dx}{\int \rho(x) dx} \\
\end{align}
$$

Why is weighted importance sampling so much better?

> Ordinary importance sampling is
unbiased whereas weighted importance sampling is biased (though the bias converges
asymptotically to zero). On the other hand, the variance of ordinary importance sampling
is in general unbounded because the variance of the ratios can be unbounded, whereas in
the weighted estimator the largest weight on any single return is one.

Problem in RL tho is that we are trying to estimate the mean return. So $\rho$ is likey to be very small!? There are a lot of possible action sequences...

$$
\begin{align}
\rho_k &= \prod_{t=k}^T \frac{\pi(s_t|a_t)}{b(s_t|a_t)} \\
\end{align}
$$

- hmm. so 'smarter' exploration policies might make it harder to do offpolicy learning?
- ok this makes sense. if there is no chance of seeing $x$ under $q$, then there is nothing we can learn about it...

## Notes

- On policy methods are a special case of off policy methods, where the target and behaviour policies are the same.
- Training off-line means you are probably also doing off-policy training. This is because while training your policy has changed, making the behaviour policy, and old version of the target policy.



TODO. Want to look into;
- https://statweb.stanford.edu/~owen/mc/. Esp the advanced variance reduction methods.
- https://arxiv.org/pdf/1702.03006.pdf
- https://arxiv.org/pdf/1509.06461.pdf
- https://arxiv.org/pdf/1811.09013.pdf
- https://arxiv.org/pdf/1811.02597.pdf
