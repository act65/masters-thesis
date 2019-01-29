What does the surface of the simplest RL problem look like!?

http://karpathy.github.io/2016/05/31/rl/

__Q:__ How does the surface change w.r.t actions versus params? How are they related?


## MNIST RL

Must generate an image. Then we get to evaluate it.
Want to generate 0-9.

How can HRL help?

Actions at pixel level seem wasteful... If I colour this pixel do I get rewarded?

$\tau (s, a) = a, \;\; r(s) = s \cdot w$

Could then go to two actions and see how that changes things!?
