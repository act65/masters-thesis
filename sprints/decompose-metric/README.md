> 1. Can we formalise what we mean by “decompose”? Can we differentiate it (so we can use it within the deep learning framework)? What is its relationship to independence criterion and independent component analysis?

TODOs

- Implement latent independence
- Disentangle sprites/moving mnist
- Derive gradients, dynamics and fixed points of $min TC(Wx)$
- Unify structural and soft priors for independence
- I would expect decomposition/disentanglement to be much easier in the RL setting as we get to take actions, to test independence!?
- Something with VQ-VAE?


***

> Disentanglement via indepencence is nice.
But, it doesnt capture modularity, abstraction, ... How can we measure and optimise them?

If they act independently then we can rewrite them as a set of modules?
$x = f_i(f_j(f_k))$.
Ok, so what if I just started out learning n different modules? Would they disentangle the latent variables?


__Q__ Cannot apply experts in parallel. Makes no sense. What property is making this the case???


$$
\begin{align}
f(g(x, z_g), z_f) &= g(f(x, z_f), z_g) \tag{commutativity} \\
\end{align}
$$

How does commutativity imply independence? How can we train for commutativity?


how would the shuffle arise in the brain? noise from info arriving at different times? cannot rely on ...!?!?

^^^ How does TC embody commutativity?!?
