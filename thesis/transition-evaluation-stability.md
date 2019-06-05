#### Evaluating the learned abstraction

> How much does approximation error in the transition function effect the evaulation of a policy?

What is the "right" way to model approximation errors from a learning algorithm?

- Is the approximation error likely to be biased in any way? Certain states or actions having more error than others? Or can we just model it as uniformly distributed noise?
- Are some of the errors likely to be correlated? Or can we sample the noise IID?  
- If so, what sort of noise? Want uniform noise on a simplex. One draw for each state.

$$
\begin{align}
\hat P &= P^{* } + E \tag{wont be row normalised}\\
&= \frac{e^{\log P + g}}{ \parallel e^{\log P + g} \parallel_{1,0}}, \; g \sim \text{Gumbel} \tag{gumbel trick}\\
&= e^gP \\
\end{align}
$$


What is the "right" way to compare the similarity between two transition functions represented in tabular form?

$$
\begin{align}
\delta &= D(A, B) \\
&= \langle A, B \rangle_F \\
&= \parallel A-B \parallel_{F}^2 \\
&= KL(? \parallel ?)
\end{align}
$$

__Proof attempt 1__

$$
\begin{align}
E &\sim \mathcal N, \parallel E \parallel_{\infty} < \delta \\
\hat P &= P + E \\
\hat V &= (I-\gamma \hat P \pi)^{-1}r \pi \\
\epsilon &= \parallel V - \hat V \parallel_{\infty}\\
&= \parallel (I-\gamma P \pi)^{-1}r \pi -  (I-\gamma (P + E) \pi)^{-1}r \pi \parallel_{\infty} \\
&= \parallel\Big((I-\gamma P \pi)^{-1} -  (I-\gamma P\pi + \gamma E\pi)^{-1} \Big)r \pi \parallel_{\infty} \\
\end{align}
$$

Want to find $X$ such that $(I-\gamma P\pi)^{-1} - (I-\gamma P\pi + \gamma\Delta\pi)^{-1} = X$. or an upper bound on $X$?

Hmph.
- Why are we inverting.
- What does the inverse do? How does it deal with small pertubations?
- https://en.wikipedia.org/wiki/Woodbury_matrix_identity. Can be derived by solving $(A + UCV)X = I$. Nice!

$$
\begin{align}
X &= (I-\gamma P\pi)^{-1}U(C^{-1}+ V(I-\gamma P\pi)^{-1}U)V(I-\gamma P\pi)^{-1} \\
\epsilon &= X r \pi \\
\epsilon[i] &\le \parallel Xr\pi \parallel_{\infty}
\end{align}
$$

What is the goal here? To write the error in terms of properties of $P, r, \pi$. The condition of $P$, the ...?

__Proof attempt 2__

How does the variance in the estimated value scale with the variance in the noise?

If I know the variance in V (wrt to properties of E) I can use this to bound the error with probability p?!?

$$
\begin{align}
y &= f(x), x\sim X \\
p(y) &= p(x) \cdot |\det \frac{dy}{dx}^{-1}| \\
E Y &= \sum_i p(y_i)y_i \\
&=  \sum_i p(x_i) \cdot |\det \frac{dy}{dx}^{-1}(x_i)| f(x_i) \\
\text{var}(Y) &= E (Y - EY)^2 \\
&= \sum_j p(y_j)(y_j - \sum p(y_i)y_i)^2 \\
&= \sum_j p(x_j) \cdot |\det \frac{dy}{dx}^{-1}(x_j)| \Big(f(x_j) - EY\Big)^2 \\
\end{align}
$$


Ok, no we have written the variance of Y wrt to X. Now we need to;
- swap f for value functional and X for noise.
- make sense of the resulting equation ...
- for a given approximation error and its likelihood $\epsilon_x, \delta_x$ we can show that the evaluation error will be less than $\epsilon_V$ with probability $\delta_V$. (approximation error is not known exactally... so must use distribution as well)



***


How does this help? What does it tell us?

- When can we trust our model?
- When can we use it to make plans?
- What guarantees can we give.
- How can I know when my model accurate enough to use it?
- I know my model is not totally accurate, but how much does this matter?

![The effect of adding noise to the transition function and then evaluating the policy under the approximately correct estimate.](../pictures/figures/noisy-transitions.png)



- How does topology (e.g. centrality, condition) effect the stability to pertubations?
- What type of pertubations? Local, adversarial, ...?
- Error in the reward doesnt seem like such a big deal? (why?!?)
