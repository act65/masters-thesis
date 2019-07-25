When designing a new algorithm for solving a reinforcement learning problem, we will want to know; are we wasting our time doing this optimisation, searching here to there and back again. Or, in ther words, does this algol converge? Another question of importance is; does my new algorithm learn faster than existing ones? Or, in other words, what is their rates of convergence, and is mine faster?

But, how can we analyse the convergence of algorithms? Analysis!

Showing an algorithm will converge, at a high level, requires two things: proof that the algorithm applies a convergence operator, and  
What does it mean to converge?


For example, a Geometric series:

$$
r \in (-1,1) \\
\frac{1}{1-r} = \lim_{n\to\infty}\sum_{i=0}^{n} r^i \\
$$

So this is an example of a contractive operator, $f(n) = x^{n}, x\in(-1,1)$.
But what if our operator is ...?

(some intuition about how it is decreasing faster than a linear rate, and because the sum is linear, ...)



## Alternative derivation of 

$$
V = (I-gamma P.pi)^{-1}r.pi
$$

For any exponentially contractive operator, acting on a field.

$f(n) = A^{n}, \det(A)\in(-1,1)$

Bellman operator.



Neumann series

$$
\begin{align}
(I -T)^{-1} &= \sum^{\infty}_{t=0} T^k \\
T &= r_{\pi} + \gamma P_{\pi} \\
???
\end{align}
$$

If the Neumann series converges in the operator norm (or in any norm?), then I – T is invertible and its inverse is the series.
If operator norm of T is < 1. Then will converge to $(I-T)^{-1}$?


- what about operators that contract at different rates?
- ?



Problems, in theory

The mean squared bellman error. We can show that the bellman operator converges wrt the infinity norm, but ... 


***

Is it possible to do an spectral analysis of these operators?
Operators = $GD, T^{* }$.


Refs

- https://en.wikipedia.org/wiki/Banach_fixed-point_theorem
- https://en.wikipedia.org/wiki/Brouwer_fixed-point_theorem
- GD convergence proof
- PI convergence proof
