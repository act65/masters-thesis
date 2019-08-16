### Linear markov decision problems (LMDPs)

> How can we construct a linear representation of an MDP?

There are a few different ways we can introduce linearity to a MDP. But, which one is best? Let's explore.

#### Linear Markov decision process (Todorov 2009)
(Exponentiated and controlling state distributions)

Define an infinite horizon LMDP to be $\{S, p, q, \gamma\}$.
Where $S$ is the state space, $p: S \to \Delta(S)$ is the unconditioned transition dynamics, $q: S \to \mathbb R$ is the state reward function an $\gamma$ is the discount rate.

How can we remove the sources of non-linearity from the bellman equation? The answer is a couple of 'tricks';

- rather than optimising in the space of actions, optimise in the space of possible transition functions.
- set the policy to be
- ?

Let's unpack these tricks and see how they can allow us to convert an MDP into a linear problem. And what the conversion costs.

\begin{aligned}
V(s) = \mathop{\text{max}}_u \Big[ q(s) -  \text{KL}(u(\cdot | s) \parallel p(\cdot | s)) +  \gamma \mathop{\mathbb E}_{s' \sim u(\cdot | s)} V(s') \Big]\\
\end{aligned}



#### Linear Markov decision process (Pires el at. 2016)
(Factored linear models)

\begin{aligned}
\mathcal R: \mathcal V \to W \\
\mathcal Q_a: \mathcal W \to \mathcal V^A \\
P(s'|s, a) = \int_w \mathcal Q_a(s', w)\mathcal R(w, s)\\
\end{aligned}

Great. But, how does this help?


\begin{aligned}
T_{\mathcal Q}w &= r + \gamma \mathcal Qw \\
T_{\mathcal R^A\mathcal Q}w &= \mathcal R^AT_{\mathcal Q} \\
T_{\mathcal Q \mathcal R} &= T_{\mathcal Q}\mathcal R \;\;\; (= T_P)
\end{aligned}


It allows us to Bellman iterations in a lower dimensional space, $\mathcal W$, rather than the dimension of the transition function.


\begin{aligned}
w^a &= T_{\mathcal R^A\mathcal Q}w \tag{bellman evaluation operator}\\
w &= M'w^a \tag{greedy update}\\
\end{aligned}


When does this matter?
Planning!! Simulating the transition function many times. Pick $\mathcal W$...

#### Linear Markov decision process (Jin el at. 2019)

\begin{aligned}
P(\cdot | s, a) = \langle\phi(s, a), \mu(\cdot) \rangle \\
r(s, a) = \langle\phi(s, a), \theta \rangle
\end{aligned}

***

So which approach is best? What are the pros / cons of these linearisations?

All of them are trying to insert some kind of linearity into the transition function.
