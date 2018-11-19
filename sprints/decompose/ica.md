## Setting

$$
x = As \\
$$

- the components $s_i$ are statistically independent
- we must also assume that the independent component must have nongaussian distributions (?!?)

There exists $C$ such that $x = AC^{−1}Cs$. Where $C$ could be a rotation, scaling, permutation. Thus we cannot uniquely recover $A$ or $s$.

## Independence

>  the components should be statistically independent. This means that the value of any one of the components gives no information on the values of the other components.

$$
p(x_1,x_2) = p_1(x_1)p_2(x_2) \\
E[f_1(x_1)f_2(x_2)] = E[f_1(x_1)] E[f_2(x_2)]  \\
$$

$$
\begin{align}
E[f_1(x_1)f_2(x_2)] &= \int\int f_1(x_1)f_2(x_2)p(x_1, x_2) dx_1 dx_2 \\
&= \int p_1(x_1)f_1(x_1) dx_1 \int p_2(x_2) f_2(x_2) dx_2 \\
&=E[f_1(x_1)] E[f_2(x_2)]  \\
\end{align}
$$

A weaker form of independence is uncorrelatedness.

$$
E[x_1x_2] - E[x_1] E[x_2] = 0 \\
$$

Uncorrelated if covariance is zero.
Independence implies zero correlation. Not the converse.

## Guassian

> the [gaussian] density is completely symmetric. Therefore, it does not contain any information on the directions of the columns of the mixing matrix A.

This argument seems less about gaussian variables and more about symmetric ones!?

> a sum of even two independent random variables is more gaussian than the original variables (via the central limit theorem -- ???)

> If the data is gaussian, it is simple to find components that are independent, because for gaussian data, uncorrelated components are always independent

Is this true for many other distributions? Uncorrelated implies independent.


Using negentropy. Why does it have to be approximated? (with kurtosis?)

## Mutual information

$$
\begin{align}
y &= Bx \tag{$s =A^{-1}x$} \\
I &= \sum H(y_i) - H(y) \\
I &= \sum H(y_i) - H(x) - log \mid \det B \mid \\
\end{align}
$$

## Resources

- [Tutorial on ICA](https://arxiv.org/abs/1404.2986)
- [Python fast-ICA](https://github.com/Felix-Yan/FastICA/)
- [White k-means -> ICA](http://proceedings.mlr.press/v32/vinnikov14.pdf)
- [FastICA example](https://github.com/Felix-Yan/FastICA)
- [ICA - Hyvarinen](https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf)
- [ICA](https://hal.archives-ouvertes.fr/hal-00417283/document)
- [Textbook - ICA](https://www.cs.helsinki.fi/u/ahyvarin/papers/bookfinal_ICA.pdf)
