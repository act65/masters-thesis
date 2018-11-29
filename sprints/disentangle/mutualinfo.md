> How can we actually compute independence?

- What is its computational complexity?
- What are practical methods? What is their error?

## Mutual information



$$
\begin{align}
y &= Bx \tag{$s =A^{-1}x$} \\
I &= \sum H(y_i) - H(y) \\
I &= \sum H(y_i) - H(x) - log \mid \det B \mid \\
\\
H(x) &= \int p(x) I(x) dx \\
I(X, Y) &= D_{KL} (p(x, y)\parallel p(x)p(y)) \\
I(X, X) &= H(x) \\
\end{align}
$$


### Total correlation

$$
\begin{align}
TC(z) &= \sum_i H(z_i) - H(z) \\
&= KL\big( p(z) \parallel \prod_i p(z_i) \big) \\
\end{align}
$$



## MMD



## Information bottleneck

> However, no algorithm is known to minimize the IB Lagrangian for non Gaussian, high-dimensional continuous random variables

hmph. someone should solve that...

When minimising $I(x, z)$, $z$ could be invariant to $x\sim p(x)$, but still be sensitive to other distributions?!?




### Latent independence

$$
\begin{align}
h &= f_{\theta}(x) \\
L_{\theta} &= -\sum_I \parallel g_i(h_I) - h_{-I} \parallel_2^2 \\
L_{\phi} &= \sum_I \parallel g_i(h_I) - h_{-I} \parallel_2^2 \\
\end{align}
$$

Try and predict the value of some latent variables from others. If the latents are independent this will not be possible.

Oh, if we set $g$ to be a linear function, then the solution would just be to set $g  = E[h_Ih_{-I}^T]$, the correlation. Thus minimising correlation... Or making orthogonal.

But how do we ensure $h$ tells us useful information about $x$? Reconstruction/generation, maximise MI, ???.

$$
E[f(x)]E[g(y)] = E[f(x)g(y)]
$$

We don't know which transforms to use, $f, g$, but instead we could search for them online - via gradient descent, parameterised as NNs. Which gives the above!?

## High order independence

Want to unify independence of objects and independence of representation.

$$
\begin{align}
TC(z) &= \sum_i H(z_i) - H(z) \\
&= KL\big( p(z) \parallel \prod_i p(z_i) \big) \\
\\
HI(z) &= \sum_i H(z_i) + \sum_{i,j} H(z_i, z_j) +\dots \sum_{i\dots n} H(z_i, \dots, z_n)- H(z) \\
&= KL\big( p(z) \parallel \prod_i p(z_i) \big) \\
\end{align}
$$

$HI$ captures __all__ the possible correlations and attempts to make them independent.
Alternatively we could do this heirarchically. Splitting the latent representation in halves and attempting to make the two halves independent.

If $z_1, z_2 \in z$ are independent, does that imply any conditions on relationship between $z_1^a, z_1^b \in z_1$? I dont think so!?

How would this naturally arise? Is there a type of noise that would encourage this? Or does energy minimisation lead to this? Independence is more energy efficient? __!!!__ that would be a nice thing to show!

## Linear


$$
\begin{align}
TC(z) &= \sum_i H(z_i) - H(z) \\
&= KL\big( p(z) \parallel \prod_i p(z_i) \big) \\
\end{align}
$$

### Total correlation

$$
\begin{align}
h &= Wx \\
L(z) &=  p_H(h)\ln(p_H(h)) - \sum_i p_{H_i}(h_i)\ln(p_{H_i}(h_i))\\
\frac{\partial L}{\partial W} &= \frac{\partial L}{\partial h} \frac{\partial h}{\partial W}  \\
\frac{\partial L}{\partial h_i} &=  \nabla p_H(h)(1+\ln (p_H(h))) - \nabla p_{H_i}(h_i)(1+\ln (p_{H_i}(h_i))) \\
\end{align}
$$


Normalsing flows to expand $p_H(h)$ in terms of $p_X(x)$
$$
\begin{align}
p_H(h) &= p_X(Wx) \mid \det \frac{\partial h}{\partial x} \mid^{-1} \\
p_{H_i}(h_i) &= ??? \\
\alpha &= \mid \det W\mid^{-1} \\
&= \alpha p_X(Wx) \ln(\alpha p_X(Wx)) - \sum_i \frac{1}{\mid???\mid}p_{X_i}(W_ix) \ln(W_ix)
\end{align}
$$

Why not do this symbolically on a computer...!>>>??!> Because we cannot actually contruct it?! Need p(x/h)...

#### Lemma 1.

Marginalise over other possibilities to calculate the probability that a single dimension is a certain value.

$$
\begin{align}
p_{H_i}(h_i) &= p(H_i=h_i, H_{-i}) \\
&= \int_{-i} p(h_{i, -i})dh_{-i} \\
&=  \int_{-i} \mid \det W \mid^{-1} p_X(W_{i, -i}x)dW_{-i} \tag{!?}\\
&=  \mid \det W \mid^{-1} \int_{-i}  p_X(W_{i, -i}x)dW_{-i} \\
\end{align}
$$

***

$$
\frac{d \det A}{dA} = \text{tr}(\text{adj}(A))
= tr(\det(A)A^{-1}) ???
$$

The above analysis will rely on square $W$. Which isnt really what we want...


***

It doesnt seem like it makes sense to only optimise $TC(h)$, where is the requirement that $h$ must contain information about $x$? Not a problem?
Because f(x) = Wx$ is deterministic, any variance in $h$ comes from $x$.

## Questions/thoughts

- Can information content be calculated by the jacobian?
- Clustering in an independent space, doesnt make sense? Should be uniformly distributed. No clusters!?
- What if the system being modelled doesnt have independent factors of variation!? (does that even make sense?)

## Resources

- [AE TC](https://arxiv.org/abs/1802.05822)
- [MMD](http://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf)
- [Estimating Mutual Information](https://arxiv.org/pdf/cond-mat/0305641.pdf)
