> Carving nature at its joints

Examples of disentanglement

## Definition

What do we mean by disentangled?

- A popular definition seems to be statistical independence.
- The latent variable to match the generating variable types. (the ability to construct sets equipped with a metric/product/transform!? mnist -> a dimension with 10 categorical variables, a ring of reals describing azimuth, bounded dims describing translation, ...)
-

## Moving mnist

With 2 digits just bouncing around they are definitely independent. But what if they are allowed to bounce off each other? Their appearance is independent but their position is now conditional.

What about having to seek out the digits on a changing scene? What about having to do arithmetic with them as well?

- want to learn some heuristics. what moves together is not independent. local relationships imply ...?

http://www.cs.toronto.edu/~nitish/unsupervised_video/
https://github.com/rszeto/moving-symbols
https://gist.github.com/tencia/afb129122a64bde3bd0c

Sprites!
https://github.com/deepmind/dsprites-dataset

## Representation

What if I have an independent variable that is a ring, or has some other topology other than a line?

What about a dataset with varying numbers of sprites/digits in the image.

## After disentanglement

Ok, so I now have a set of independent latent variables, $z$. What should I do now?

Imagine: I want to learn a classifier using the disentangled representation, $g(z)$. The classifier learns to only use a subset of the latent dimensions, $S \subset Z$. So this means that the classification is invariant to changes in the ignored dimensions.

How can this knowledge of invariance be used!?
What about equivariance? $f(T(x)) = T(f(x))$?
A correlation between the target and an independent dimension implies equivariance!?

Learn $T_i$ such that

***

problem. TC only works wth balanced training data!?

## Alternative

Want a way to optimise disentanglement that isnt not derived from Bayes/Variational. $\parallel z_p - e(x)\parallel_1$. >>> Noise as targets!?

A possible alternative route?
- [NAT](https://arxiv.org/abs/1704.05310)
- [Clustering via NAT](https://openreview.net/pdf?id=BJvVbCJCb)
- [Sinkhorn AEs](https://arxiv.org/pdf/1810.01118.pdf)


## Temporal disentanglement

> We propose an unsupervised variational model for disentangling video into independent factors, i.e. each factor’s future can be predicted from its past without considering the others.

http://willwhitney.com/assets/papers/Disentangling.video.with.independent.prediction.pdf

Great, we can model objects and their trajectories independently. But what is there is some long range dependencies/correlation?

$$
E_{z\sim q_{\phi}(z\mid x)} - D_{KL}(q_{\phi}(z\mid X) \parallel p_{\theta}(z_1)) + \sum_{t=2}^n E_{z_{t-1}\sim q(z_{t-1}\mid z_{t-2}, x)} D_{KL}(q_{\phi}(z_t\mid z_{t-1}, x)\parallel p_{\theta}(z_t\mid z_{t-1})) \\
$$

Can this model actually be used as a generative model? Can generate a trajectory given an init, but can it generate a digit?

$$
\begin{align}
\mathop{\text{min}}I(z_t^i, z_{t+1}^j) \\
\end{align}
$$

Could construct with independent modules applied to the $i$th dimension. But want the ability to learn to prune the unneeded weights.

Does $\mathop{\text{min}}I(z_t^i, z_{t}^j) \implies \mathop{\text{min}}I(z_t^i, z_{t+1}^j)$?

***

Want a linear decomposition of the transition function so we can use feature expectations.

$$
\begin{align}
s_{t+1}^i = f(s_t^i) \tag{wwhitney} \\
s_{t+1}^i = s_t^i +  \\
\end{align}
$$

$$
\begin{align}
R(s) &= w^T \phi(s)\\
V^{\pi}(s) &= w^T\mathbb E [\sum \gamma^i\phi(s_i)] \\
&= w^T \mathbb E [\Big(\phi(s_i) + \gamma \mu^{\pi}(\tau(s_i,a_i))\Big)] \\
R(s,a) &= w^T \phi(s, a)\\
Q^{\pi}(s, a) &= w^T \mathbb E [\sum \gamma^i\phi(s_i, a_i)] \\
&= w^T \mathbb E [\Big(\phi(s_i, a_i) + \gamma \mu^{\pi}(s_{i+1},a_{i+1})\Big)] \\
\end{align}
$$


## Thoughts

- What happens if I learn a disentangled representation in domain A. And then want to transfer it to domain B? What if a new feature correlates with two of the independent features?

## Resources

- [Beta VAE](https://arxiv.org/abs/1804.03599)
- [Structured disentangled representations](https://arxiv.org/abs/1804.02086)
- [ICM](https://arxiv.org/abs/1712.00961)
- [Information dropout](https://arxiv.org/abs/1611.01353)
- [Emergence of invariance and disentanglement](https://arxiv.org/abs/1706.01350)
