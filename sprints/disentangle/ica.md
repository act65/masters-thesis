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
