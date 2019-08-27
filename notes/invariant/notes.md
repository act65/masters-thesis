
What is the geometry of this?
- Have the Q values being pulled towards their true values by the tradtional TD error, or similar.
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
Want a measure of distnace that is low if we can find a simple (linear) transform that takes $x\to y$. And high if $x\to y$ requires lots of non linearity. Kinda.

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
