# Research proposal

## Introduction

Transfer learning for reinforcement learning.

### What is reinforcement learning (RL)?

An easy setting to understand RL is in the Markov decision process (MDP) setting.

A MDP is defined as a tuple, $\{\mathcal S, \mathcal A, P(s_{t+1} \mid s_t, a_t),R(s_t, a_t, s_{t+1})\}$. Where $\mathcal S$ is the set of possible states (_for example arrangements of chess pieces_), $\mathcal A$ is the set of actions (_the different possible moves, left right, diagonal, weird L-shaped thing, ..._),  $P(s_{t+1} \mid s_t, a_t)$ is the transition function which describes how the environment acts inresponse to the past and to your actions (_in this case, your opponent's moves, and the results of your actions_), and finally, $R(s_t, a_t, s_{t+1})$ is the reward function, (_whether you won (+1) or lost (-1) the game_), which you are trying to maximise.

This setting is easily generalised to other, more interesting, applications. For example, if we weaken the requirement on that the learners observation $x_t$ is fully describes the state $s_t$, then we could apply this framework to partially-observable decision processes, such as [StarCraft II]() or self-driving cars.

(originally inspired by research in physcology? Thus 'good' actions can be reinforced by receiving positive rewards, while 'bad' actions can be punished.

It should be noted that the problems that make RL hard are; "trial-and-error search and delayed rewards" [Sutton and Barto]. Unlike supervised learning, which gives the learner feedback (_I think that digit is a 5 -> no, its a 6_), in RL the learner only receives evaluations (_I think that digit is a 5 -> wrong_).

### What is transfer learning (TL)?

An easy setting to understand TL is in the multi-task setting (but there are others, one-shot learning, zero-shot learning, ...??? and even distribution shift could be viewed as transfer learning between the past and present)

Imagine we have two classification tasks, $A, B$, which each consist of pairs of observations and targets $\text{task} = \{(x_i,t_i) : i \in [1:N]\}$. Now, a learner, $f$, trained on task $A$ and achieves a loss when evaluated on task $A$. We denote this as: $\mathcal L^A(f_A)$ (subscript denotes training task, superscript denotes evaluation task).

We say transfer has occured when $L^A(f_A) > L^A(f_{B\to A})$ (goal is to minimse loss). That is to say, that training on $B$, and then training on $A$ imporves performance on $A$. This is also known as [pretraining](). Similarly, we could transfer in the opposite direction, $L^A(f_A) > L^A(f_{A\to B})$. Training on $A$ and then training on $B$ improves performance on the original task, $A$. (note this is actually quite hard [ref!?]())

An example of transfer is !!!

But there is more to it than that? Forward vs backward transfer. Many applications, ...
Needs to be safe!
What about efficient transfer? Increasing the speed or learning (aka meta learning).


Captures the idea of learning something more abstract about the similarities between two distinct tasks. Closely related to the notion of generalisation.

### Why do we care?

Applications. Existing research. Blah Blah.

## Transfer as decomposition

One way to achieve transfer learning is to: decompose complex phenomena into a set of modules/atomic-factors. Or in other words, disentangle independent factors. (need refs!!!)




Even if we have a great decompostion of a complex phenomena, there is a lot of meta information required to actually use this decomposition. Which modules do what? How are they related? When should I apply a given module?

__Q__ is decomposition necessary for transfer?


Note, there are other approachess to transfer(?). Learning a metric? Distillation? EWC? MDL regularisation? ???

For example;
- take a symmetry group, can be rewritten as the composition of a single atom with various transformations. Thus we have decomposed the symmetry into is parts.
- Aka finding the basis.
- Decomposition of objects and relations?



So, the question becomes: how can we efficiently learn a useful decomposition?
There are a few approaches to this question;
- which loss function should be used? what is the training objective? (which leads to ICA, meta learning, )
- how can we structure our learner so that a decomposition is a necessary result? Modular nets, ... (uses domain knowledge to help design networks)
- how is structure and symmetry involved?

The belief that complexity can be decomposed, that it canbe built from smaller/few parts, is at the heart of all unsupervised learning. For example, it is comon to assume that there are a set of latent variables that combine (non)linearly to generate the observed complexity.

Two ways to enforce a decomposition; structural constraints or a regulariser (?). (other ways?)

We can build a decomposition into the strucutre of our model, for example, we can define a transition function that optimises next step prediction error, and a policy that is optimised via its correlation with reward. Another example could be ensembles, where to get a decomposition of the learners we can feed them different inputs, forcing them to do different things. (this requires domain knowledge!?)

Or we can regularise a general function approximator to build disentangled representations. For this we needa a differentiable measure of disentanglement.

### Decompositions in RL

Let's explore some decompositions in RL.

Model-based Rl can be viewed as the decomposition of a reinforcement learner into a transition function and a policy. This facilitates transfer as, the learner can be given a new task, requrining a new policy, (while remining in the same environment) and simply reuse its model. Notably model-free learners cannot do this. (refs!) If we could get model-based RL to work, it would take use a long way to solving many problems in XXX?. But, in the real world, our ability to learn and transfer models is still an unsolved problem. Which will require a large amount of transfer itself.

There are other examples of existing work attempting to use a decomposition to facilitate transfer, for example;

- [Feudal networks]() decompose the policy into a manager and a worker who work a different time scales.
- [Learning to learn]()
- [Modular meta-learning](https://arxiv.org/abs/1806.10166)

#### Measure

How do we know if two phenomena should be decomposed? What measures are there that tell use the relationship between two variables?

Define decomposition: ?
Define disentangled representation: ?

hmm. Hypothesis, we are missing a coherent definition that captures our intuitive notion of decomposition/disentanglement.

#### Modular ML

- Modular QA approches?


#### Heirarchies

One of the most important ideas being used in machine learning is that of a heirarchy. The idea of composing simple conecpts into more complex ones, ...

- Heirarchy of value functions (long term planning)
- Heirarchy of actions ()
- ?

#### Ensembles

[Outrageously large networks](https://arxiv.org/abs/1701.06538)

### Decomposition as reductionism

Gush about science! Automated science.


## Setting

Not only do we want to achieve transfer learning, but we want to do it in realistic settings. This requires ...
While there has been some good work (as highlited above) in transfer RL. It has often only been applied in restricted settings.
Half of the work to make transfer RL truly useful will be to extent the settings that existing RL approaches can succeed in.

One of the reasons, in my opinion, that the machine learning field has progressed quickly is its use of shared benchmarks on currently out-of-reach problems. [ImageNet]() is a great example of this for the computer vision community, and the [Atari ALE]() is currently the canonical benchmark for the RL community.

While the Atari ALE has been a great benchmark, and there is still progress to be made (what progress???). It is starting to age;

- the Atari ALE has been solved, in the sense of achieving super-human performance on all of the games (citiation!?).
- continuous actions
- partial info
- ?

But! Need cleverly designed benchmarks that dont take thousands of CPU/GPU hours to evaluate on.

What is just out of reach now? What benchmarks exist to measure progress?


We want to be able to transfer knowledge in the most general setting;
- continuous actions (especially important for robotics),
- resources constraints (online learning),
- partial information,
- ?

Toy problems to solve/datasets?
- learn periodic table from chemistry data (?)
-

## Proposed research

Goal of research is ??? understanding! How to get understanding? A necessary part is building it and showing it work.  

### A benchmark

The field lacks a clear benchmark for decomposition. ???
Find and design toy problems!

Show that existing approaches cannot do X.

## ?

First step, learn a model. Then use it to plan.

There are many challenges to learning a good model. Efficient and safe exploration (or transfer from a simulation),

Overview of existing approaches to decomposing complexity
- heirarchical RL,
- model-based RL,
- meta learning,
- modular ML,
- ?




### Specific ideas to explore

Questions (these may be ill-posed, trivial, or solved, but hopefully I will find out soon...)

Decompositions

- Linear Markov decision problems
- Does a temporal decomposition (moving averages at different scales) of rewards produce a generalisation of meta learning?
- Model the transition function as a mixture of densities, $s_{t+1} = \mathop{\text{argmax}}_{s_{t+1}} \prod_i p_i(s_{t+1} \mid s_t)$.
- ?

Fundamental RL

- What can you learn from an interactive environment (you can take actions and observe their results) that you cannot learn from a (possibly comprehensive) static dataset? Similarly, what can you learn when you have access to an $\epsilon$-accurate model, that you cannot learn from an interactive environment?
- Why does [distributional RL]() provide such a large advantage?
- Efficient search. Relationship between tree search and gradient descent? Local and global optimisation.
- Dynamic programming and its relationship to diffusion
- Temporal credit assignment???

Unsupervised learning

- How can we learn decomposed representations? ICA, Lateral inhibition, residuals?
- Inverse energy learning. Similar to [inverse reinforcement learning]() what if we assume that the observations we make are the results of an energy being minimised, $\Delta x = -\eta\frac{\partial E}{\partial x}$. Thus we could try to learn $E$.
- MLD? symmetry strucutre?  What are the 'right' priors? How can we optimise them?

Model-based learning (with partial information)

- Build a [differentiable neural computer]() with locally structured memory. (2d and then generalise?)
- Is the ability to localise oneself necessary to efficiently solve partial information decision problems?
- In model-based learning the model must learn its our approximation of the policy being followed. This seems wasteful, how can we avoid this?
- Off-policy correction for curiosity. The exploration policy may influence the dynamics observed, that needs to be corrected.

Planning with a learned model (with continuous actions)

- If a model is being learned online how can we efficiently update value estimates computed using the old model?
- How can you backpropagate gradients through the argmax functions required for planning?
- Exploitable models
- If I am using an imperfect learned model to generate plans, how can I ensure that I do not plan for 'fantastic' outcomes (aka they are fantasies).

Generalisation/transfer!?!?

- ?


I will not attempt to explore all of these. Rather I will take random walk through them, attempting to make progress where possible. The goal will always be to; show that the proposed problem is actually a problem, design the minimal viable experiment to test new ideas, test new ideas...

### Proposed deliverables

- Interactive tutorial on ???
- Model-based learner that is performant in the Atari ALE (or similar)
- Essay on the future of model-based RL
- Definition and construction of a new benchmark that is currently out-of-reach,
- Implementation of ??? for the tensorflow community

Hopefully a few papers, but that is conditional on making discoveries.
