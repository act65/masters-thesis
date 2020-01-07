Have some uncertainty about the transition fn and the reward fn.
These might be empirical distributions based on data observed so far.

$$
p = P(\tau[s, a, s'], s, a, s'| D) \\
p = P(r[s, a], s, a| D) \\
$$

For a given $\tau, r$, we can solve for $Aut(\tau, r)$ using

How does $Aut(\tau, r)$ change with changes in $\tau, r$?

(ahh. will need approximate symmetries first? actual symmetries will be too sensitive to changes in $\tau, r$)


***

How to find all subgroups of a group.
If we can do so efficiently, then we want to order them by 'complexity'!?
Want to infer from just a couple of observations, not access to P / r.

***

Nodes = states, edges = m(W_{Ps}(s, s'), W_{Rs}(s, s')).
What about the actions!?!? Not clear to me how this graph encodes the similarity between actions, the $g_s(a) = a'$.

Should be Nodes = state-actions?


***


\subsubsection{Inferring symmetries in data with priors}

Pick a simpler setting. No noise. Discrete domain. No action.
First we need to be able to identify group structures from observations of ???.
Then, we can generalise to observations of the groups action on another set.
Then, we can generalise to noisy observations.


> Ok, what data do we need to be able to infer a group structure $(Z_2, S_4, A_3, Di_8)$?

- Some vs all elements (if we need all, then isnt really an inference problem...).
<!-- (although, observing all the group elements might not be so bad, when they act on a large set!?) -->
- Pairs $c = a \circ s$ vs triples $c = a \circ ?$ (where do the triples come from?)

What information could be provided, or needs to be inferred?

- Number of elements in the group.
- The $n\times n$ relations.
- The type of group (cyclic, alternating, Sporadic) \href{https://en.wikipedia.org/wiki/Classification_of_finite_simple_groups}{ref}
- The identity of subgroups. $Z_2 \times X = \text{Obs}$

Under which constraints?

1. Identity
2. Inverse
3. Closure
4. Commutative / symmetric

***

- Impossible without triples. TODO prove.
- If we are given triples, then we have the job of matrix completion. That needs to satisfy [1,2,3,4].
- We can form triples from pairs when if we know that the transformations are linear?!
- ?

\subsubsection{Cayley completion}
% Expected to find something on the net about this. matrix completion of cayley tables. Am I thinking about it wrong?

Ok. We guessed a $n$ (or it has been given) and now we have an incomplete cayley matrix: we filled it in with some observations.

Q What is our earch space of possible cayley matrices? How large is it?
How does a new piece of data reduce the number of possible cayley tables?
