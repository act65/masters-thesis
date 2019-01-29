Why would we want more depth? (greater length of time deps?? which is related to state size!?)

Equivalence between subgoals and options.

$$
g \in f(\mathcal S) \\
\omega \in g(\mathcal A^k) \\
$$

What is the relationship between these two spaces?

A good set of options should "cover" the important parts of state space. That is, all reachable states, using the given options, should cover the subgoals.

$\mathcal R = \{s': \exists \omega_{0:t}, s' =\tau(s, \omega_{0:t})\}: \;\;f(\mathcal S)\subseteq \mathcal R$.



### Pros/cons?

Subgoals

- Needs access to a metric or a model
- Must be solved (via MPC or a learned policy)
-

Options

- Easy and cheap to execute (just look up from memory and go)
- Discovery is !?!
- Is rigidly fixed. Cannot change lower level options else learning must start over!?
- ?


Want:
- system that starts with subgoals. But if the same things is computed/optimised many times, turn it into an option.
- variable sized options!?
