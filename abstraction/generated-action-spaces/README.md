> Want: A way to generate toy puzzles / envs for exploring HRL's to reduce learning complexity.

## Motivation

It does not seem clear (at least to me) what we mean by Hierarchical RL.

To study emergent languages, or HRLs advantages over RL, or temporal credit assignment we need some toy problems. What does a toy HRL problem look like? What are it properties?

Let's generate some envionments that have structured but complex action spaces.

Want to be able to generate large action spaces with a large amount of structure.
E.g. a thousand dimensional action space, that maps to only (say) 4 possible actions(up down left, right).


## Experiment

> Generalising between structured action spaces

1. Edit [coinrun]()/similar to have different action spaces (yet they might share some - heirarchical - structure).
2. Benchmark openAI [baselines]() and see how well the perform with;
  - transfering between spaces that share structure
  - large(er) action spaces
3. construct abstract-action learner that out performs the baselines...

Ok. So what properties might the present in the mapping from abstract actions to ground actions?

- Invertible (just a rotation or permutation)
- Heirarchical
- ?


## Interfaces

We have an environment, an interface and an agent.
The agent must learn to use the interface to act in the environment.

An interface is a mapping from the agents actions to actions within the environment.
$$I: A_{agent} \to A_{env}$$
For example; a mouse an keyboard for interfacing with a computer, a steering wheel and foot pedals for interacting with a car, etc...

But;
- the interface might add unnecessary complexity to the action space.  
- the interface might change or break.
-

Want to;
- adapt quickly to a new, similar interface.
- generalise and still achieve goals despite a break / change in the interface.
- compress the complexity within the interface's action space
- ?


***

Pieter Abbeel mentioned an env they designed.
> Grid world where agent must discover the passcodes for the actions (left right up down). For example left might be 0,0,1.
