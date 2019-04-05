For a long time, when I was approaching HRL I was trying to figure out how it was better then RL.
Obvious (in hindsight) but not so short answer is: it isn't 'better'. That is because of the no-free-lunch theorem.

Simply split the learning complexity between discovering a higher level language.
And the using it.

> The challenge in the single-task case is overcoming the additional cost of discovering the options; this results in a narrow opportu- nity for performance improvements, but a well-defined objective. In the skill transfer case, the key challenge is predicting the usefulness of a particular option to future tasks, given limited data.
Combined. (Konidaris 2019)


$$
E_{f\in F}\Big[E_{x\sim A}[ f(x)]\Big] = E_{x\sim B}[ f(x)] \\
$$

Refs

- [No Free Lunch Theorems for Optimization](https://ti.arc.nasa.gov/m/profile/dhw/papers/78.pdf)
- [A conservation law for generalization performance](http://dml.cs.byu.edu/~cgc/docs/mldm_tools/Reading/LCG.pdf)
- [No More Lunch: Analysis of Sequential Search](https://acff25e7-a-62cb3a1a-s-sites.googlegroups.com/site/boundedtheoretics/CEC04.pdf)
- [A No-Free-Lunch Theorem for Non-Uniform Distributions of Target Functions](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.71.8446&rep=rep1&type=pdf)
- [Simple Explanation of the No-Free-Lunch Theorem and Its Implications](https://link.springer.com/content/pdf/10.1023%2FA%3A1021251113462.pdf)

If you find a search algorithm that performs worse than randomly on some set of optimisation problems, you know it must perform better on the set if all other optimisation problems.
(not quite? as we are not allowed rnd search algols here?)

> Focusing on generalisation rather than accuracy has the effect of normalising differences in traniing set size. The conservation law tells is that overall generalisation performance remains null for every n. As a consequence, performance will increase with increasing n for some regualrities only to the extent that it decreases with n for others.

(can I simulate this empirically. an example!? of decreasing performance?)



- Want to explore a proof where the cost functions have bounded complexity (in the information theorietic sense. they are compressible. Maybe we pick all f such that their information is <= k). Does NFL still hold? (related [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=870741) and [this one - bounded info](http://mattstreeter.org/Research/mstreeter_gecco_2003.pdf)?)

## Matching

> Empirical success in generalisation is always due to problem selection.

We have a matching problem that needs to be solved. Problems to optimisers.
What makes this matching problem hard?

- It can be hard to etimate the problem from calls to the oracle... If we could do so then we could easily optimise it.
- ???


## Local search algorithms (Gradient descent)

Are gradient descent algols a part of this framework??
Grads require knowledge of $f$?! (well AD does)
But, not necessarily, could just evaluate twice!? f(x) - f(x+dx).
