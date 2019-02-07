> Want: A way to generate toy puzzles / envs for exploring HRL's to reduce learning complexity.

## Motivation

It does not seem clear (at least to me) what we mean by Hierarchical RL.

To study emergent languages, or HRLs advantages over RL, or temporal credit assignment we need some toy problems. What does a toy HRL problem look like? What are it properties?

Let's generate some envionments that have structured but complex action spaces.

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

Want to be able to generate large action spaces with a large amount of structure.
E.g. a thousand dimensional action space, that maps to only 4 possible actions(up down left, right).


Pieter Abbeel mentioned an env they designed.
> Grid world where agent must discover the passcodes for the actions (left right up down). For example left might be 0,0,1.



***

__NOTE:__ Here we have only considered the symmetries of the transition function in action space. But what we really care about is the return (at least in model-free RL). Thus we can get more compression / abstraction by compressing actions-states that achieve the same return. (can take the high road or the low road. They arrive in different locations by different paths. Yet they recieve the same reward, thus... ? Not sure about this)

The reward fn might

- be a fn of a subspace of state space?
