
What is the geometry of this?
- Have the Q values being pulled towards their true values by the traditional TD error, or similar.
- Have 'similar' state-actions being pulled together.
  - In the case the $\chi$ is based on Q values. This would mean a kind of density based attractive force!?
  - Higher density positions are more attractive.

***

Relationship to off policy RL?!?

***
__TODOs__. Extend this first order analysis to;
- momentum. What is $\dot y$ if we update the parameters with momentum?
- natural gradient?
- ?

***

Want a measure of distaace that is low if we can find a simple (linear) transform that takes $x\to y$. And high if $x\to y$ requires lots of non linearity. Kinda.

***
1. Let $\phi$ be learned via an adversarial value fn.
2. Could use this representation to construct a kernel: $k(s, s') = \langle \phi(s), \phi(s') \rangle$
3. Could use kernel to couple gradients. $\dot f = -\eta KH\nabla \ell$

Doesnt make sense if we are learning a value function? Why not just use $\phi$ as a representation for the value fn? What is gained from the coupling versus just using the representation to predict the value?

$V(\phi(s))$ vs $\dot V = -\eta K_{\phi}H\nabla \ell$
Might even be the same thing?!?

***

[A generalized representer theorem](https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=FF17282B5B5DAAB20E1CFD46D3A77B05?doi=10.1.1.42.8617&rep=rep1&type=pdf)

***

We think we have found a mirror symmetry. We use this to reduce the search space, and only explore one side. But we were wrong about the mirror symmetry, and there is a sparse, large, reward in one of the sides. Did we do worse than before?


Questions
- where does compression come into disentanglement?!
- Completeness of the abstraction. it can caputre any / all symmetries? which symmetries are harder to learn?

Related

- [The Information Bottleneck Method](https://www.cs.huji.ac.il/labs/learning/Papers/allerton.pdf)
- [Generalization Error of Invariant Classifiers](https://arxiv.org/abs/1610.04574)
- [On Learning Invariant Representation for Domain Adaptation](https://arxiv.org/abs/1901.09453)
- [?](?)
- Fisher preconditioning

TODOs
- Test clustering with unparameterised $\chi$
- Test clustering and generalisation with parameterised $\chi$
- Test ?!?


***

Approximate symmetries!!!

***

- How can symmetries be discovered?
- How well do NNs discover symmetries?
- How can we add priors into NNs to help them learn quicker?
- How can we learn transferable representations?

## Data augmentation

There exist domain specific strategies for; images, audio, text, ... what about RL?

For images we have;
For audio we have;
For text we have;  ...

More generally, there exist techniques like

- [AutoAugment](https://arxiv.org/abs/1805.09501)
- [Population Based Augmentation](https://arxiv.org/abs/1905.05393)

that, provided some transforms, find the best ones to use.

***

- [Pairing Samples](https://arxiv.org/abs/1801.02929)
- [mixup](https://arxiv.org/abs/1710.09412), [Manifold Mixup](https://arxiv.org/abs/1806.05236), [Between-class Learning](https://arxiv.org/abs/1711.10284)
- [MixMatch](https://arxiv.org/pdf/1905.02249v1.pdf)

### for RL.

> To force exploration in strategy space, during training (and only during training) we randomized the properties (health, speed, start level, etc.) of the units, and it began beating humans. [OpenAI Five](https://openai.com/blog/openai-five/)

> The rules of Go are invariant to rotation and reflection. This fact was exploited inAlphaGoandAlphaGo Zeroin two ways. First, training data was augmented by generating 8 symmetriesfor each position.  Second, during MCTS, board positions were transformed using a randomlyselected rotation or reflection before being evaluated by the neural network, so that the Monte-Carlo evaluation is averaged over different biases. The rules of chess and shogi are asymmetric,and in general symmetries cannot be assumed.AlphaZerodoes not augment the training dataand does not transform the board position during MCTS. [AlphaZero](https://arxiv.org/abs/1712.01815)

- [HER](https://arxiv.org/abs/1707.01495)


- The usual image augmentation. Noise, rotations, translations, occlusions?, colour, hue, saturation, ...
- State augmentation:
- Action augmentation:
- Reward augmentation:
- Value augmentation:
- Policy augmentation:

Off policy evaluation!!!?!??

## How efficiently do NN discover symmetries?

They dont discover them, they only see them and remember.
They dont generalise well...
Want a regulariser that makes their outputs group like?!?

## How to (efficiently) discover symmetries?

- Does it require meta-learning?
- priors?
- ?

A measure of complexity?

***

Two trajectories / policies are similar because a similar amount of risk is taken.
p(V) = p(V).

$$
\int \int p(\zeta, f_1)p(\zeta, f_2)R(\zeta, \gamma))d\zeta d\gamma \\
$$


***

RL mixup. ???

$$
V_{\pi_1 + \pi_2}(s) = V_{\pi_1}(s) + V_{\pi_2}(s)
$$


### Approximate symmetries

What do we do when;
- $f(T(x)) = f(x)$ for most $x$?
- $f(T(x)) \approx f(x)$
-


***


If we have a set of invariant units and we compose them together. Is the resulting construction still invariant to the same things?

[Tensor network decompositions in the presence of a global symmetry](https://journals.aps.org/pra/pdf/10.1103/PhysRevA.82.050301)
***
Subgoals.
Partition the space.
All within a partition are similar because they are attempting to achieve the same goal.
Partitions may be similar in a few senses?
- They share the same actions.
- Policies used to achieve another goal, B, can be used to also achieve


***

Dataset is a collection of $(s, a, r){_i}$ s with some return, $R(i, \gamma)$.

Imagine them as a large graph. Different discounts induce different edge weights. $(s, a, r){_i} \sim (s, a, r){_i}$ if $\parallel R(i, \gamma)-R(j, \gamma)\parallel$.


***

Symmetries are about taking the topology / graph of the observed data and reducing it to its quotient.



***


1) symmetry of architectures; 2) symmetry of weights; 3) symmetry of neurons; 4) symmetry of derivatives; 5) symmetry of processing; and 6) symmetry of learning rules
https://arxiv.org/pdf/1712.08608.pdf


***

How are symmetries in the value fn related to projecting the value polytope!?
***

$$
Q(\pi, s, a, \gamma, g) = \mathop{\mathbb E}_{\zeta \sim d(s, a, \pi)} [R(\zeta, \gamma, g)] \\
Q^{* }(s, a, \gamma, g) = \mathop{\mathbb E}_{\zeta \sim d^{* }(s, a)} [R(\zeta, \gamma, g)] \\
Q^{* }(s, a, \gamma, g) = Q^{* }(T(s, a, \gamma, g)) \\
\tau(s, a) = s', r \\
$$

***

What happens if you give an agent a basis of action extended by some repitions? An over complete basis. L, R, LL, RR, LLL, RRR, ...?
Which actions does it use?

## Measuring similarity for RL.

If we restrict our attention to only similarity via the return, there are still a few ways to implement this.

Are their;
- trajectory (/ ordering) of rewards 'similar'?? $\sum_t \gamma^t\mid r_t - r'{_t} \mid$ (no. this doesnt really work. we need to find the closest trajectories between s, s'. or use all of them?!)
- hyperbolic discounting 'similar'? $\sum_{\gamma} | V^{\gamma}(s) - V^{\gamma}(s')|$
- aversarial value fns 'similar'? $\parallel\phi(s) - \phi(s')\parallel$
- expectations 'similar'? $V(s)-V(s')$
- distributions 'similar' (via some IPM) $\mathop{\text{min}}_{f} f(x) - f(y)$
- statistical moments 'similar'? $\sum E[(R(s) - R(s'))^p]$


Why would we want to choose one over the other?
__TODO__ Experiments with each measure of similarity.
Uses these measure of similarity to group states / actions and see which matches what we want.
__WANT__ Some nice ways to visualise.

***

RL is a type of supervised learning. So we can hope to achieve disentantlement!??
