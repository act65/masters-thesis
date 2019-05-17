How does HRL actually help?
Why should we care?

Want to prove that HRL can give exp speed up of learning complexity over RL.

## Definitions

__QUESTION:__ What do we mean by structured but complex action spaces?

- symmetries in action space: $\exists \;T \; s.t.\; \tau(s, a) = \tau(s, T(a)),\; a \neq T(a)$
  - Low dimensional structure in action space: $\tau(s, a) = f(s, g(a))$ where $g(a): \mathbb R^n \to \mathbb R^m$, $n >> m$
  - Low dimensional structure in time: $\tau_k(s, a_1, a_2, \dots a_k) = f(s, g(a_1, a_2, \dots a_k))$ where $g(a_1, a_2, \dots a_k): A^k \to \Omega$, $g^{-1}(\Omega) \subset A^k$

What about multiple symmetries?

- In time: Where we migh have $\tau_k(s, a_1, a_2, \dots a_k) = f(s,  a_1, a_2, \dots a_k)$ and $f_k(s, \omega) = f(s, h(\omega_1, \omega_2, \dots \omega_k))$. (__Q:__ but if this more abstract symmetry exists, then why do we care about the lower level one!?)
- what if they symmetries only exist in time AND dimension? Ie there are no symmetries just in time, or just in dimension!? Is that possible!?
- $f(x, y) \neq f(T(x), y) \neq f(x, T(y)) = f(T(x, y))$. $\tau(s=0, a=0) = \tau(s=1, a=1)$. Walking backwards from facing ths start is equivalent to walk forwards from facing the finish.


__TODO:__ What about approximate symmetries!? How can these be generated!?

__NOTE:__ What if $g$ was also a fn of $s$? Symmetries based on the state-action pair?! $\tau(s, a) = \tau(T(s, a))$

***

Want to show that;

- accurate: action abstraction can recover (within $\epsilon$) the optimal solution.
- efficient: the complexity of finding the optimal solution (compared to the original) is lower?

But these seem totally banal!?

Want a tool that takes a MDP, and a representation (of the action space - or state space?) and derives/estimates bounds on its accuracy and complexity (relative to the optimal ground solution).
(could then use this for training... oh!)

# Questions

- Want to understand (visualise) how the state-transition reachability graph changes as a function of different action abstractions.
- How does the action abstraction transform the value fn?
- And how temporal abstraction can integrate with action abstraction.
- Can we construct temporal and state abstractions from action abstractions?
- Is there always an optimal set of abstract actions?
- What propreties of the original MPD can we predict from only looking at sequences of (optimal) abstract actions?

### What are the fundamental problems with abstraction for RL?

- Stability of the heirarchy? (obv for HRL)
- Transfer (knowing when and how well)
- how do sparse rewards effect learning an abstraction?
- want the ability to tailor our design of an abstract to the current problem!? (this turns into meta learning)
- how does memory come into this? If our goal is to solve optimisation problems as efficiently as possible, it seems sensible to learn an abstraction, and to remember past problems that have been abstracted. <- problem with this. Online adaptation of the abstraction.
- ?

### What are the things we think we understand, but dont? Or dont realise are important.

- ?

### What dont we understand about actions?

- Action centric representations, are relative?!How do these compare to the absolute state representations. Which one is better?
- ?

## Possible settings/approaches

 - Original problem -> abstraction -> abstract optimal policy -> difference between abstract and original optimal policies
 - Explore the loss surface of a linear RL problem and explore how it gets transformed under an abstraction
 - Frame as finding a set of spanning edge types for a graph. (?)

***

Specifically, want to study the generalisation of (abstract) actions.
How can this be done?

Actions define changes to the state.
- Two action sets might span the same space of changes but be represented in a different basis.
- Two action sets might $\epsilon$ span the same space of changes in different bases. How does that effect the difference between their performance?
- Two action sets might have vastly different spans, yet when combined temporally, they span the same k-step action-transition space.

^^^ This is about what the action spaces can represent. Not their internal structure!?

## References

- [Towards a Unified Theory of State Abstraction for MDPs](http://anytime.cs.umass.edu/aimath06/proceedings/P21.pdf)
- [Near Optimal Behavior via Approximate State Abstraction](https://arxiv.org/abs/1701.04113)
- [State Abstractions for Lifelong Reinforcement Learning](http://proceedings.mlr.press/v80/abel18a.html)
- [On the Complexity of Solving Markov Decision Problems ](https://arxiv.org/pdf/1302.4971.pdf)


## Tools

The tools for analysing the learnability of X are ??? crappy.

How can I accurately estimate learnability and generalisation?


***

Problem that occurs in POMDPs. Am I correctly modelling the state?
You want to learn what action a does. So you do $\tau(s, a)$ over many $s\in A$. But the effect of $a$ correlates with the subset of states $A \subset S$ yo are experimenting in.
For example. Balls always fall towards the gound (if you test only on earth).


***

Ideate and pick 4 more sprints.

- [ ] Unsupervised options (how good can random ones be!?)
- [ ] Equivalence of goal/option conditioned value fns
- [ ] Build a three (or even better, N) layer heirarchy
- [ ] Explore how different approaches scale (computational complexity) in the number of layers in the heirarchy
- [ ] Use a learned reachability metric to measure proximity to subgoals (and thus use to give rewards)
- [ ] A heirarchical subgoal net that uses MPC rather than learned policies
- [ ] Explore function approximation for options a = f(w) (rather than look up table)
- [ ] How does this related to a decomposition of the value function?
- [ ] How to achieve stable training of a hierarchy?
- [ ] Filtering / gating state space to the lower levels
- [ ] Connection to evolution of language.
- [ ] The benefit of a heirarchy of abstractions? (versus use a single layer of abstraction). Transfer!?
- [ ] Design a new programming language. Learner gets access to assembly and must ??? (build a calculator? allow a learner to build websites easy? ...?). What would be the task / reward fn? (should be easy to learn to use, require few commands to do X, ...?)
- [ ] A single dict with the ability to merge, versus a heirarchy!?
- [ ] What is the relationship between abstraction and generalisation!?

***

> 1) What do we mean my abstraction? Let's generate some RL problems that can be exploited by a learner that abstracts.

> 2) How does abstraction actually help? Computational complexity, sample complexity, ... when doesn't it help? When it is guaranteed to help?

> 3) Can we learn a set of disentangled actions. How does that help?

> 4) How can we use an abstraction to solve a problem more efficiently? Use MPC + abstraction. Explore how different abstractions help find solutions!?


***

- Relationship between tree search and HRL? (divide and conquer for MPC) Recursive subgoal decomposition.  https://arxiv.org/pdf/1706.05825.pdf
- Absolute versus relative perspectives (relationship to subgoals and options)


***

**

> 5.Build a differentiable neural computer (Graves et al. 2016) with locally structured memory (start with 1d and then generalise to higher dimensions). Is the ability to localise oneself necessary to efficiently solve partial information decision problems? Under which conditions does the learned index to a locally structured memory approximate the position of the agent in its environment.


Memory structures + Locality.

- https://www.nature.com/articles/nn.4661.pdf
- https://arxiv.org/pdf/1602.03218.pdf
- https://arxiv.org/pdf/1609.01704.pdf
- DNC


***

An alternative view on disentanglement. But what is its relationship to independence?

- Counterfactual estimates of gradients for modules?
- Blackbox jacobian sensing


We want n different modules that specialise in their tasks? How can we learn different specialists?

- winner takes all credit assignment. positive feedback/lateral inhibition/...?
- give access to different inputs/resources. they physically/computationally cannot do the same thing...
- train on different tasks...
- ?


***

Ok, imagine we give two specialists different inputs (say, left-right halves of mnist) and train them separately on the same classification task. Their outputs are __not__ going to be independent... The fact the the left half classifies its input as a 2, makes it likely that the right half will also classify its input as a 2.

So, in this instance, the two modules will output independent results if;

- they are trained on the same task yet receive independent inputs
- the two experts are trained on different tasks, which are independent
- ?

***

Questions
- __Specialisation__! Independently useful contributions.
- __Q__ How is independence related to decomposition?

## Resources

- [MoE](https://arxiv.org/abs/1701.06538) <-- could measure the hidden states MI/TC!?
- [DDP for Structured Prediction](https://arxiv.org/pdf/1802.03676.pdf)



***

## Alternative derivation of V = (I-gamma P.pi)^{-1}r.pi

For any exponentially contractive operator, acting on a field.

> Geometric series

$$
r \in (-1,1) \\
\frac{1}{1-r} = \lim_{n\to\infty}\sum_{i=0}^{n} r^i \\
$$

So this is an example of a contractive operator, $f(n) = x^{n}, x\in(-1,1)$.
But what if our operator is ...?

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
