---
title: model-based rl
---
#Model-based RL

Why do we want to do this? Motivation! What problem does model-based RL solve?

Want to find certain settings where model-free > model-based and vice versa.

Number of samples $n$, a measure of the complexity of the environment $k$, the complexity of the model $N$, ...??? (what else?)

$$
d(\nabla \pi, G_{\text{model-based}}) = poly(n, k, N) \\
d(\nabla \pi, G_{\text{model-free}}) = exp(n, k, N) \\
$$

problem is that this ignores the actual optimisation (it could be that some types of inaccuracy in the gradient estimation actually aid learning...)

Is there a case where a model of the dynamics naturally emerges?

## Definition

??? A reinforcement learner that does not have an explicit model of the transition function.
Struggling with this definition. Best I can do is related to how the learning is supervised.

Model-free only gets access to the reward. Model-based also uses the state-action trajectories.
(I want to show that this extra learning signal can change performance from exp to poly)

> conventional wisdom holds that model-free methods are less efficient but
  achieve the best asymptotic performance, while model-based methods are
  more efficient but do not produce policies that are as optimal
  ([TDM](https://bair.berkeley.edu/blog/2018/04/26/tdm/))

## A fundamental trade-off

We need to sample reality... If not we can end up planning a fantasy.

(when failing is cheap it can be easier to try, make a bunch of errors, rather than plan. esp with innaccurate models...)

In the case where the environment is more complex than can possibly be modelled, then ...?
Planning isnt going to work... But, what advantages can be gained from an partial model?

In the case where the task is very simple. If light is red, push button.
Dont need a complex model of the environment...

what about models that capture high level/deep principles? While inaccurate (in their ability to predict a future state, they ...???)

It depends on how expensive it is to sample the real reward!?
If queries are cheap/unrestricted, then let's just do model-free RL!?
How can we assign cost to queries to the oracle? Are there different types of cost?

- memory/size of policy
- calls to the oracle
- reward received

***

This is really just a question of gradient estimation.
What is the variance/bias of the estimator?

Want to see proof that Q-learning and PG have zero bias. Want to bound their variance.
(is that all we care about!? -- Is the gradient pointing in the right direction? Does it have the right magnitude?)

How could this be studied? Construct a differentiable model and check the accuracy of various estimators.

Dont actually care if $\tau (s_t, a_t) = \hat \tau (s_t, a_t)$ we care if $\nabla \tau (s_t, a_t) = \nabla \hat \tau (s_t, a_t)$.
Is there a way to evaluate the gradient of the transition function?

***

(In the unconstrained memory case)
__Cojecture:__ Model-based learning is the optimal solution to model-free learning

I can imagine a model-free system learning to learn to use future inputs as targets to learn a model!!?!
If we used a very large RNN to model $Q(s_t, a_t)$, it could correlate the difference between past and future state and actions, thus ...?

Model-free methods must learn some estimate of future dynamics!? (how can this be shown? neurons that correlate with dynamics?)
How much of the future dynamics does a model-free method learn? How much is necessary?

## Resources

- [TDM](https://bair.berkeley.edu/blog/2018/04/26/tdm/)
- [Successor features](https://arxiv.org/abs/1606.05312)
- [Model-free planning](https://openreview.net/forum?id)
