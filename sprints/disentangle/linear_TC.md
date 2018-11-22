
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
&= \Big( p_X(Wx) \mid \det \frac{\partial h}{\partial x} \mid^{-1} \Big) \ln(p_X(Wx) \mid \det \frac{\partial h}{\partial x} \mid^{-1}) \\
& \quad\quad- \sum_i p_{H_i}(h_i)\ln(p_{H_i}(h_i)) \\
\alpha &= \frac{1}{\mid \det W\mid} \\
&= \alpha p_X(Wx) \ln(\alpha p_X(Wx)) - \sum_i \frac{1}{\mid???\mid}p_{X_i}(W_ix) \ln(W_ix)
\end{align}
$$



doesnt work!? cant calculate the det of a vector...
$$
\sum_i \Big( p_{X}(W_ix) \mid \det \frac{\partial h_i}{\partial x} \mid^{-1} \Big)\ln(p_{X}(W_ix) \mid \det \frac{\partial h_i}{\partial x} \mid^{-1})
$$

Marginalise over other possibilities to calculate the probability that a single dimension is a certain value.

$$
\begin{align}
p(H_i=h_i) &= \int_{-i} p(h_{i, -i})dh_{-i} \\
&=  \int_{-i} \mid \det W \mid^{-1} p_X(W_{i, -i}x)dW_{-i} \\
&=  \mid \det W \mid^{-1} \int_{-i}  p_X(W_{i, -i}x)dW_{-i} \\
\end{align}
$$

$$
\frac{d \det A}{dA} = \text{tr}(\text{adj}(A))
= tr(\det(A)A^{-1}) ???
$$

The above analysis will rely on square $W$. Which isnt really what we want...
