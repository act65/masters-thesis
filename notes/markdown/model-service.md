---
pagetitle: Serving models as a service
---

<div>
# Model based RL

I am trying to imagine the future of model-based RL: model-based RL as a service (?).

Imagine you are an enterprising individual. You have some problems you want to solve;

- generating new drugs for curing cancer,
- a room that needs cleaning,
- explaining matriods,
- driving,
- poverty.

What service would help you solve them? Well, if you had an accurate, efficient model of the world, you could plan for the outcomes you want. For example, with a model of quantum physics, chemistry and human biology we could search through possible drugs for ones that will cure cancer(s).

I can imagine two types of service, pertrained models and explorers.

## Pretrained models

Could include ones for;
- generalised chemical reactions,
  - human scale robots (indoors / outdoors). Classical physics,
  - human learning,
  - road rules, human behaviour,

Given a domain specification (the model of robot, the sensors, a high level description of its task)
we supply a model of internal and external dynamics.

For example, a customer might request
The model is able to identify things like; water, glass,

(but what about online learning, and the ability of the planner to exploit the model -- various resolutions of modelling!?)


Question: why would anyone trust us? Our guarantee on a model would be ??? on a set of open benchmarks?

  ```python
  import world_models as wm
  import mcts

  domain.config = {
      'actions': [],
      'observations': [],
      'reward': fn
  }

  model = wm.build(
      config=domain.config
  )

  planner = mcts.Planner(
      transition_fn=model,
      action_space=domain.action_space,
  )

  planner = planner.rollout(10000)
  # can now roll the model out to the real world


  # will need to adapt the model if its predictions are incorrect.
  # ?
  ```

  What about on the service side?

  Want a meta learned fn that takes configs in and returns models designed
  for that domain.


  ```python
  request = requests.watch('https://...')
  if request:
      observations, actions, reward = parse_config(request.config)
      design_model(observations, actions, reward)
  ```

  Will need a massive amount of transfer.

  Imagine we can separately model physics and human behaviour and we are tasked with designing
  a new model for self-driving cars. How can this work?

  encoder(s): observations -> distribution over entities
  transition_fn is constructed from many modules
  f o g o h: state -> state

  How can these models be composed meaningfully?
  When is it ok to apply them independently?
  f(state) + g(state) -> state_t

  Need both the encoder from physics and from human behaviour to map into the same latent space!?!?!?

## Explorers
  - learns a model of a new environment
  - does experiments (possibly breaks things...)

(only has observations and actions. no reward or goal.)

Want the learner to come with as many relevant priors as possible. To speed up the learning.

This will be closer to meta learning.

You put the explorer in your environment, and set back, well back. If quickly constructs a model of its environment.

- allowed to mix chemicals, (should produce the periodic table, ...)
- interact with a computer (...)
- ??? ->

## Marketplace

What about model marketplace?

Consists of;
- models
- benchmarks
- rewards

People pay for the use of each.


Benchmarking could become a big business.
- Have your models (say driving robots) independently verified by a trusted service. (will need common physical targets)
- ?



Problem is the protection of IP.



</div>
