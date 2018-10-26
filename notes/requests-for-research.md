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

<!-- ## Continuious options

https://arxiv.org/pdf/1703.00956.pdf -->

</div>
