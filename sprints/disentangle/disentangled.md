> Carving nature at its joints

What do we mean by disentangled? A popular definition seems to be statistical independence.
Want. And orthogonal/indpendent basis.

> disentanglement (Bengio et al., 2013) each learned feature in Z should represent structurally different aspects
of the observed phenomenas (i.e. capture different sources of variation).

## Motivation

What problem does this solve?
Imagine you are receive information in a basis that is not disentangled. How does this make learning harder?
Can test empirically. But how can it be proved?

Must assume that the generating process uses n independent variables to generate the observations.

For example, consider mnist. We could model it as the output of a single variable. The variable codes for one of the 60,000 images. Or we could model it as 4 independent variables, the number, the width, the slantedness, the height.

Forget dimensionality reduction for a second. Imagine we have an indpendent basis, $A$, and a basis that is not independent, $B$, such that $f(A) = B, f^1(B) = A$. Which one is better for learning $X = g(A)$? Is it always true that $L(g(A)) > L(g(B))$? (no) as maybe $g = f$. How does the relationship between the basis and the target fn effect learning?

If a subset of dims is needed for a classification task, all that is needed to pick the right subset of features is to calculate correlation.(? no)

## Independence

>  the components should be statistically independent. This means that the value of any one of the components gives no information on the values of the other components.

$$
p(x_1,x_2) = p_1(x_1)p_2(x_2) \\
E_{x_1, x_2\sim p}[f_1(x_1)f_2(x_2)] = E_{x_1\sim p_1}[f_1(x_1)] E_{x_2\sim p_2}[f_2(x_2)]  \\
$$

***

If $X, Y \in [0, 1]$ are independent then we can have $(0, 0), (0, 1), (1, 0), (1,1)$. But we must have them with a uniform distribution, otherwise it would be possible to guess the value of $Y$ from $X$. If we know that $(1, 0)$ occurs with zero probability, and we receive $X=1$, then we can guess with high confidence $Y=1$.

So, for independence, there must be maximum entropy over all pairings!? What about triplets, and higher order correlations? Wait a minute. What I just said is that we want the join distribution, $p(x, y)$ to have maximum entropy

Problems.
- The requirement for each pair to have the same frequency seems unwanted. Is there a way to weaken this? Dont think so, this is required for independence!?
- The above intuition doesnt capture the separability of the different variables. Does it?

***

Not sure independence is what we want. Take the CelebA dataset. If we constrain every latent dimension to be independent, and one of the dimensions is a smile vector, then that precludes any other dimension from altering the mouth. As the prescence of a yawn vector would explain away the smile vector. Would create a collider which makes independent variables dependent when the effect is observed.

Take Marcus' example. Faces and lighting. There are two sources of variation, the face present, an the lighting used when the image was recorded. But, lighting and faces are not independent!? No wait. They are independent, but when observed they can explain each other away!? No problem here? If we optimise the latent representation to have independent factors, it is coherent to want those factors to represent lighting/face. (???)
Obviously, if lighting=0, then I cant see anything... And while it is true that the lighting doesnt effect the face that is actually there, it does effect the image recorded. Hmm.


***

A lot of important things are not independent. For example, the velocity of an object is not independent of its shape/orientation/position (in the toy problems we analyse this isnt true). But its velocity is independent of its color, ??? (not that much?).

## After disentanglement

Ok, now I have a set of independent latent variables, $z$. What should I do with them?

Imagine: I want to learn a classifier using the disentangled representation, $g(z)$. The classifier learns to only use a subset of the latent dimensions, $S \subset Z$. So this means that the classification is invariant to changes in the ignored dimensions.

How can this knowledge of invariance be used!?
A correlation between the target and an independent dimension implies equivariance!?

For a transformation, $T$, $e(T(x)) = T(e(x))$.
Or rather. Let $z = e(x)$. Then apply some transform, $T$, to $x$. $z_T = e(T(x))$. If $e$ has learned to embed $X$ in a disentangled representation $Z$ then $\exists i: z[i] \neq z_T[i], \forall j \neq i: z[j] = z_T[j]$. That is, a single dimension describes the transformation applied.

Ok, now imagine we want to learn a classifier that is invariant to $T/z_i$. We can:
- learn $f({e(x)}_{-i})$ (apply $f$ on a subset of the features)
- learn $\sum_i g(x_i)$ (how can you share over that dim!?)

We need to construct the equivalent of extract patches!?

## Alternative

Want a way to optimise disentanglement that isnt not derived from Bayes/Variational. $\parallel z_p - e(x)\parallel_1$. >>> Noise as targets!?

A possible alternative route?
- [NAT](https://arxiv.org/abs/1704.05310)
- [Clustering via NAT](https://openreview.net/pdf?id=BJvVbCJCb)
- [Sinkhorn AEs](https://arxiv.org/pdf/1810.01118.pdf)




## Basis

$$
\begin{align}
\forall x_i \in X \\E[f(x)]
f^* , y^* = \mathop{\text{argmin}}_{f, y} D(x_i, f(y)) \tag{$y \in Y, f \in F$}\\
s.t. \mathop{\text{argmin}}_{Y, F} \mid Y\mid + \mid F \mid\\
\end{align}
$$

Imagine;
- $y_i$ is an ordered set of elements, say symmetries, $\sigma, \tau$, and $f$ is the composition of those elements.
-

What we want is to find a basis which we can efficiently use to construct observations.

Imagine;
- a set of images. We can search through combinations of the basis set to find a mixture that approximates $x_i$.

## Modular

> Disentanglement via indepencence is nice.
But, it doesnt capture modularity, abstraction, ... How can we measure and optimise them?

If they act independently then we can rewrite them as a set of modules?
$x = f_i(f_j(f_k))$.
Ok, so what if I just started out learning n different modules? Would they disentangle the latent variables?


__Q__ Cannot apply experts in parallel. Makes no sense. What property is making this the case???

Whitney + Fergus apply the transforms in parallell!? Can apply in parallell if we are applying different transforms to independent objects. But cannot apply independent transforms on a single object in parallel.
Heirarhical indepencence? Two sets of distributed representations and are intra/inter independent.


$$
\begin{align}
f(g(x, z_g), z_f) &= g(f(x, z_f), z_g) \tag{commutativity} \\
\end{align}
$$

How does commutativity imply independence? How can we train for commutativity?


how would the shuffle arise in the brain? noise from info arriving at different times? cannot rely on ...!?!?

^^^ How does TC embody commutativity?!?

#### Temporal disentanglement

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

## Interventional robustness

> Concretely, we call a causal process disentangled when the parents of the generated observations do not affect each other (i.e. there is no total causal effect between them)

Need a metric!
- the mutual information of a single latent dimension Zi with generative factors G1, . . . , GK where in the ideal case each Zi has some mutual information with one
generative factor Gk but none with all the others.
- train predictors (e.g. lasso or random forests) for a generative factor Gk based on the representation Z. In a disentangled model, each dimension Zi is only useful (i.e. has high feature importance) to predict one of those factors.

> we assume the generative factors themselves to be confounded by (multi-dimensional) C, which can for example include a potential label Y or source S. Hence, the resulting causal model C → G → X allows for statistical dependencies between latent variables Gi and Gj , i 6= j, when they are both affected by a certain label, i.e. Gi ← Y → Gj . However, a crucial assumption of our model is that these latent factors should represent elementary ingredients to the causal mechanism generating X (to be defined below), which can be thought of as descriptive features of X that can be changed without affecting each other (i.e. there is no causal effect between them).

Aka

$$
\forall i\neq j: \int \mid \frac{\partial g_i}{\partial g_j}(x) \mid dx = 0
$$

(the equation can only be zero if all $\frac{\partial g_i}{\partial g_j} = 0$ -- reminds me of a trick used in variational calc!?)

Wait a minute. Need to be careful here. This is just a linear relationship, and is equivalent to corrrelation? But we want no dependence at all, so all higher order derivatives must also be zero? $\frac{\partial^2 g_i}{\partial g_jg_k}$. Actually no, it is ok, as we have integrated over all $x$. So we dont need to consider the higher order grads.

## Emergence of disentanglement

A natural product of

- energy minimisation. It is cheaper to process infomation independently of other info
- noise!?

***

Two random variables X and Y are independent if and only if the characteristic function of the random vector (X, Y) satisfies

$$
\begin{align}
\varphi_{(X, Y)}(t, s) &= \varphi_{X}(t) \cdot \varphi_{Y}(s) \\
\varphi_{X}(t) &= E[e^{itX}] \\
\end{align}
$$

hmm.

## Questions/thoughts

- problem. TC only works wth balanced training data!?
- What happens if I learn a disentangled representation in domain A. And then want to transfer it to domain B? What if a new feature correlates with two of the independent features?
- __Q__ How does explaining away come into this!?

## Resources

- [Beta VAE](https://arxiv.org/abs/1804.03599)
- [Structured disentangled representations](https://arxiv.org/abs/1804.02086)
- [ICMs](https://arxiv.org/abs/1712.00961)
- [Interventional robustness](https://arxiv.org/abs/1811.00007)
- [Information dropout](https://arxiv.org/abs/1611.01353)
- [Emergence of invariance and disentanglement](https://arxiv.org/abs/1706.01350)
