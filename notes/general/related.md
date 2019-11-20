\hypertarget{model-based-rl}{%
\section{Model-based RL}\label{model-based-rl}}

Pros and cons.

Model-based learning can be bad\ldots{} There may be many irrelevant
details in the environment that do not need to be modelled. A model-free
learning naturally ignores these things.

The importance of having an accurate model!

For example, let \(S\in R^n\) and \(A\in [0, 1]^n\). Take a transition
function that describes how a state-action pair generates a distribution
over next states \(\tau: S \times A \to \mathcal D(S)\). The reward
might be invariant to many of the dimensions.
\(r: X \times A -> \mathbb R\), where \(X \subset S\).

Thus, a model mased learner can have arbitrarily more to learn, by
attempting to learn the transition function. But a model-free learner
only focuses on \ldots{}

This leads us to ask, how can we build a representation for model-based
learning that matches the invariances in the reward function. (does it
follow that the invariances in reward fn are the invariances in the
value fn. i dont think so!?)

Take \(S \in R^d\) and let \(\hat S = S \times N, N \in R^k\). Where
\(N\) the is sampled noise. How much harder is it to learn
\(f: S \to S\) versus \(\hat f: \hat S \to \hat S\)?

\cite{Wang2019a,Kaiser2019}
