We have two random variables, $X, Y$. I want to know whether it is easier (in terms of sample complexity) to;

 - estimate their means
 - decide whether $X = Y$

(with high confidence / low error)

***

> Why do I care?

We can accelerate learning by sharing knowledge between 'similar' RVs.
(We know image $x$ and image $y$ are similar, therefore give $x$'s labels to $y$, and vice versa -- this assumes the labelling is a noisy process making an image a RV.)

But, how do we know whether two RVs are similar.

We estimate their mean / distribution (or something else?), and compare (??). Now we can use this knowledge to share data (following our image example: share past / future labels).

__But__. We gain nothing from sharing now, as we already know their mean / distribution.

What I want to know is; is it possible to infer that two RVs are similar before we have a good estimate of their mean / distribution?

Note. We are not considering the case where knowledge of similarity can generalise. We are just trying to understand whether there can be an advantage of ...

***

In RL we care about whether the mean (of the value fn) is the same.
Hmm. Seems harder to know whether the distributions are the same.
But then what are we grouping based on.

The similarity measure, $\chi_A$, used for sharing must be the same as the similarity measure, $\chi_B$, used to deciding whether $X=Y$. (?)
$\chi_B$ must be coarser or the same as $\chi_A$? If that is true, then (1) doesnt seem possible???
(Only want to share data between things we know are similar...)

How could temporal abstraction make this work?

What about as a way to guide exploration. These things might be similar. Double check?! How to capture this with min of uncertainty?

***

Consider a guassian setting. Where $X, Y$ are gaussian distributed random variables,
with $\mu_x, \mu_y, \sigma_x, \sigma_y$.

We take $n$ samples $\{x_0, x_{n-1}: x_i \sim \mathcal N(\mu_x, \sigma_y)\}$ from $X$ and
We take $m$ samples $\{y_0, y_{m-1}: y_j \sim \mathcal N(\mu_y, \sigma_y)\}$ from $Y$.

$$
\begin{align*}
\epsilon_{\mu} &= \parallel \mu_x - f(D_x) \parallel + \parallel \mu_y - f(D_y) \parallel \\
&= f(n, \sigma_x) + f(m, \sigma_y) \\
\end{align*}
$$

***

Unrelated. Kinda.

$$
\chi \nabla \ell(x) \\
$$

Rather than sharing updates. As we might be wrong about some of the similarities. Share when predicting. $\chi$ is a matrix.
Current estimate of $V_{s=i}$. But rather use $(1-\chi(i, j))V_{s=i} + \chi(i, j)\big(\frac{V_{s=i}+V_{s=j}}{2}\big)$. Or more generally (which isnt the same as the above... need to normalise the matrix!?).

$$
\pi(\cdot |\chi \cdot s_t)
$$
