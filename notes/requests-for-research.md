---
pagetitle: Requests for research
---

<div>

<p>To stay sane I need to write down some of the _actionable_ ideas that occur to me.
Otherwise I have the tendency to hoard them.
So, these are the questions I am not going to answer (argh it hurts!).
They appear to be perfectly good research directions, but "you need to focus" (says pretty much everyone I meet).</p>

# Requests for research

_(the number of stars reflects how open the problem is:, 1 star means little room for interpretation, 3 stars mean that there are some complex choices to be made)_

__Controlled implicit qualtiles__ &#9734; Extend [Implicit quantile RL](https://arxiv.org/abs/1806.06923) (which works suprisingly well) to use [control](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.43.7441&rep=rep1&type=pdf) [variates](https://arxiv.org/abs/0802.2426).

__Atari-onomy__ &#9734; Make a [taskonomy](http://taskonomy.stanford.edu/) of the [Atari](https://gym.openai.com/envs/#atari) games, showing how 'similar' each game is to others.

__Who is the best critic?__ &#9734; &#9734; Which value fn approximation has the best inductive bias suited to learning value functions? A value function is the expected future return of a given state. $V(s_t) =\mathbb E\big[ \sum_{i=t}^T \gamma^{T-t+i} R(s_t) \big]$. We can approximate the value function with a parameterise function, but which one? [Relational neural networks](https://arxiv.org/abs/1706.01427), [decision trees](https://en.wikipedia.org/wiki/Decision_tree), [neural processes](https://arxiv.org/abs/1807.01622), ... . Learning value functions have a couple of issues, large class imbalance/rare events, online training,  distribution shift, ... .

__Learner discrimination__ &#9734; &#9734; &#9734; Just by observing a player learn, can we identify the learning algorithm is it using to learn? For example, can we distinguish the learners in [OpenAI's baselines](https://github.com/openai/baselines/), PPO, AC2, AKTR, ...?

__Unbiased online recurrent opptimisation__ &#9734; &#9734; Can we extent [UORO](https://arxiv.org/abs/1702.05043) to the RL setting and the continuous learning setting by combining it with an online version of [EWC](https://arxiv.org/abs/1612.00796)?

__Meta learning emerges from temporal decompositions__ &#9734; &#9734; [Meta-RL](https://arxiv.org/abs/1611.05763) trains a learner on the aggregated return over many episodes (a larger time scale). If we construct a temporal decomposition (moving averages at different time-scales) of rewards and aproximate them with a set of value functions, does this produce a heirarchy of meta learners? (will need a way to aggregate actions chosen in different time scales, for example $\pi(s_t) = g(\sum_k f_k(z_t))$)

__The learning complexity of small details for RL__ &#9734; &#9734; In [Model based learning for atari](https://arxiv.org/abs/1903.00374) they learn a model using self-supervision. They study the model and show that in some cases it misses small (yet extremely important) details, such as bullets. How much easier does learning these details become when we have access to correlations with rewards, rather than just a reconstruction error?

__Optimsable values__ &#9734; &#9734; &#9734; When working with continuious actions, it is a hard problem to calculate the action that yields the max value $\mathop{\text{argmax}}_a Q(s, a)$, especially when $Q$ is a large non-linear neural network, then solving the equation below can be hard. What if we could do sgd on $Q(s, a)$, or learn $Q(s, a)$ that is convex wrt a?

<!-- - Relationship between tree search and HRL? (divide and conquer for MPC) Recursive subgoal decomposition.  https://arxiv.org/pdf/1706.05825.pdf -->

<!-- __Visualise and understand the loss surface of simple RL problems__ &#9734; &#9734; -->

<!-- ## Continuious options

https://arxiv.org/pdf/1703.00956.pdf -->

<!-- A spectrum between accurate/fast models and slow/accurate ones.
How can we bootstrap one model from others?
Reverse, local-global interactions, accuracy mask, time step, ... -->


</div>
