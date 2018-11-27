# Independence

>  the components should be statistically independent. This means that the value of any one of the components gives no information on the values of the other components.

$$
p(x_1,x_2) = p_1(x_1)p_2(x_2) \\
E_{x_1, x_2\sim p}[f_1(x_1)f_2(x_2)] = E_{x_1\sim p_1}[f_1(x_1)] E_{x_2\sim p_2}[f_2(x_2)]  \\
$$

This seems like a very unusual property!? It is weaker than orthgonoality. As if $f_1, f_2$ were orthogonal then $E[f_1(x_1)f_2(x_2)] = 0$ (actually that is not be true!?!).

$$
\begin{align}
\langle f_1, f_2 \rangle &= \int f_1(x)f_2(x)dx \\
&= 0 \tag{orthogonal}
\end{align}
$$

proof
$$
\begin{align}
E[f_1(x_1)f_2(x_2)] &= \int\int f_1(x_1)f_2(x_2)p(x_1, x_2) dx_1 dx_2 \\
&= \int p_1(x_1)f_1(x_1) dx_1 \int p_2(x_2) f_2(x_2) dx_2 \\
&= E[f_1(x_1)] E[f_2(x_2)] \\
\end{align}
$$

A weaker form of independence is uncorrelatedness.

$$
E[x_1x_2] - E[x_1] E[x_2] = 0 \\
E[(x_1- E[x_1])(x_2-E[x_2])] = 0 \\
$$

Uncorrelated if covariance is zero.
Independence implies zero correlation. Not the converse.

***

Two random variables X and Y are independent if and only if the characteristic function of the random vector (X, Y) satisfies

$$
\begin{align}
\varphi_{(X, Y)}(t, s) &= \varphi_{X}(t) \cdot \varphi_{Y}(s) \\
\varphi_{X}(t) &= E[e^{itX}] \\
\end{align}
$$

hmm.

***

Invariance!?

$$
P(A) = P(A \mid B) \\
$$

Nope. Not what we want!?
__Q__ How does explaining away come into this!?

***

Wait a minute. Blind source separation. Does ICA even work for BSS? If we assume they are from independent sources then ...?

Would produce a box, not two orthogonal lines?

***

If $X, Y \in [0, 1]$ are independent then we can have $(0, 0), (0, 1), (1, 0), (1,1)$. But we must have them with a uniform distribution, otherwise it would be possible to guess the value of $Y$ from $X$. If we know that $(1, 0)$ occurs with zero probability, and we receive $X=1$, then we can guess with high confidence $Y=1$.

So, for independence, there must be maximum entropy over all pairings!? What about triplets, and higher order correlations?

Problem. The requirement for each pair to have the same frequency seems unwanted. Is there a way to weaken this? Dont think so, this is required for independence!?

Problem. The above intuition doesnt capture the separability of the different variables. Does it?

Wait a minute. What I just said is that we want the join distribution, $p(x, y)$ to have maximum entropy


***

Not sure independence is what we want. Take the CelebA dataset. If we constrain every latent dimension to be independent, and one of the dimensions is a smile vector, then that precludes any other dimension from altering the mouth. As the prescence of a yawn vector would explain away the smile vector. Would create a collider which makes independent variables dependent when the effect is observed.

Take Marcus' example. Faces and lighting. There are two sources of variation, the face present, an the lighting used when the image was recorded. But, lighting and faces are not independent!? No wait. They are independent, but when observed they can explain each other away!? No problem here? If we optimise the latent representation to have independent factors, it is coherent to want those factors to represent lighting/face. (???)
Obviously, if lighting=0, then I cant see anything... And while it is true that the lighting doesnt effect the face that is actually there, it does effect the image recorded. Hmm.

## ICA - Setting

$$
x = As \\
$$

- the components $s_i$ are statistically independent
- we must also assume that the independent component must have nongaussian distributions (?!?)

There exists $C$ such that $x = AC^{−1}Cs$. Where $C$ could be a rotation, scaling, permutation. Thus we cannot uniquely recover $A$ or $s$.

### Guassian

> the [gaussian] density is completely symmetric. Therefore, it does not contain any information on the directions of the columns of the mixing matrix A.

This argument seems less about gaussian variables and more about symmetric ones!?

> a sum of even two independent random variables is more gaussian than the original variables (via the central limit theorem -- ???)

> If the data is gaussian, it is simple to find components that are independent, because for gaussian data, uncorrelated components are always independent

Is this true for many other distributions? Uncorrelated implies independent.


Using negentropy. Why does it have to be approximated? (with kurtosis?)

### Whitening

The whitening transformation is always possible. -> components are uncorrelated and variances equal 1.

$$
E[xx^T] = USV^T \\
\hat x = US^{\frac{1}{2}}V^Tx \\
E[\hat x \hat x^T] = I
$$
???

> Here we see that whitening reduces the number of parameters to be estimated. Instead of having to estimate the $n^2$ parameters that are the elements of the original matrix A, we only need to estimate the new, orthogonal mixing matrix $\hat A$. An orthogonal matrix contains $\frac{n(n − 1)}{2}$ degrees of freedom.

### ICA -> Perceptron -> PCA -> linear AE

$$
y = f_{\theta}(x) \\
\mathop{\text{min}}_{\theta} \log \frac{p(y)}{\prod_i p(y_i)} \\
\frac{\partial L}{\partial \theta} = ??? \\
$$

- What are the dynamics of minimising this loss? What is the geometry?
- Do we really need to decorrelate W at every time step?

### Resources

- [Tutorial on ICA](https://arxiv.org/abs/1404.2986)
- [Python fast-ICA](https://github.com/Felix-Yan/FastICA/)
- [White k-means -> ICA](http://proceedings.mlr.press/v32/vinnikov14.pdf)
- [FastICA example](https://github.com/Felix-Yan/FastICA)
- [ICA - Hyvarinen](https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf)
- [ICA](https://hal.archives-ouvertes.fr/hal-00417283/document)
- [Textbook - ICA](https://www.cs.helsinki.fi/u/ahyvarin/papers/bookfinal_ICA.pdf)


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

!!! [Estimating Mutual Information](https://arxiv.org/pdf/cond-mat/0305641.pdf)

### Total correlation

$$
\begin{align}
TC(z) &= \sum_i H(z_i) - H(z) \\
&= KL\big( p(z) \parallel \prod_i p(z_i) \big) \\
\end{align}
$$

[AE TC](https://arxiv.org/abs/1802.05822)

## MMD

http://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf



## Latent independence

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

## Questions/thoughts

- Can information content be calculated by the jacobian?
- Clustering in an independent space, doesnt make sense? Should be uniformly distributed. No clusters!?
- What if the system being modelled doesnt have independent factors of variation!? (does that even make sense?)
