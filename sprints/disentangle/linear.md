
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
