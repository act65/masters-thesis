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


Want a tool that takes a MDP, and a representation (of the action space - or state space?) and derives/estimates bounds on its accuracy and complexity (relative to the optimal ground solution).
(could then use this for training... oh!)

# Questions

- Want to understand (visualise) how the state-transition reachability graph changes as a function of different action abstractions.
- How does the action abstraction transform the value fn?
- And how temporal abstraction can integrate with action abstraction.
- Can we construct temporal and state abstractions from action abstractions?
- Is there always an optimal set of abstract actions?
- What propreties of the original MPD can we predict from only looking at sequences of (optimal) abstract actions?
- Action centric representations, are relative?!How do these compare to the absolute state representations. Which one is better?

### What are the fundamental problems with abstraction for RL?

- Stability of the heirarchy? (obv for HRL)
- Transfer (knowing when and how well)
- how do sparse rewards effect learning an abstraction?
- want the ability to tailor our design of an abstract to the current problem!? (this turns into meta learning)
- how does memory come into this? If our goal is to solve optimisation problems as efficiently as possible, it seems sensible to learn an abstraction, and to remember past problems that have been abstracted. <- problem with this. Online adaptation of the abstraction.
- ?

Actions define changes to the state.
- Two action sets might span the same space of changes but be represented in a different basis.
- Two action sets might $\epsilon$ span the same space of changes in different bases. How does that effect the difference between their performance?
- Two action sets might have vastly different spans, yet when combined temporally, they span the same k-step action-transition space.

## References

- [Towards a Unified Theory of State Abstraction for MDPs](http://anytime.cs.umass.edu/aimath06/proceedings/P21.pdf)
- [Near Optimal Behavior via Approximate State Abstraction](https://arxiv.org/abs/1701.04113)
- [State Abstractions for Lifelong Reinforcement Learning](http://proceedings.mlr.press/v80/abel18a.html)
- [On the Complexity of Solving Markov Decision Problems ](https://arxiv.org/pdf/1302.4971.pdf)
