Inductive bias in exploration.

Different exploration strategies might prefer to explore some areas before others.

How can we quantify an exploration strategy?

- calculate the steady-state distribution over states. No.

the important point is the order of exploration.
- first need to show that one exploration stragety has a prefered ordering.
- then can compare different exploration strategies.

$$
P(s, t) = d^{\pi} [s = s_t]
$$

__Observation:__ If for all t, the entropy is maximised, then there is no bias.

$$
\forall t, H(P(s, t)) = 1/n
$$

__QUESTION:__ Do some exploration strategies have a bias in the limit of $t\to \infty$?

$$
H(P(s, \infty)) \neq 1/n
$$

For any exploration strategy, a requirement should be that it conveges to the max entropy state-distribution.


***

Wait, what are we exploring?
Policies? Or states?
States.
We definitely do not want to explore all possible policies...

***

What priors could make sense in exploration?

Want to explore first;
- states that we can control,
- states that allow us to control many other dimensions of the state
- states that behave the most unpredictably (have large variance).
  or the opposite? states that we can be certain about.
- state-actions with large effect. a large change in state

***

How does this make sense?
- in discrete spaces. likely to have a bias towards more central nodes? the ones that are 'easy' to reach.
- how does inductive bias make sense in cts spaces? exploring some dimensions more than others. but we can find another basis when that might not be true?!

***

exploration strategies

- optimisim in the face of uncertainty
- intrinsic motivation
  - surprise (rewarded for seeing unpredictable states)
  - novelty (rewarded for seeing new states)
- count based
  - and pseudo count?
- max entropy

***

- note. but we dont just care about exploring states??? inductive bias in state-actions?
