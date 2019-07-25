---
pagetitle: Serving models as a service
---

<div>
# Model based RL

I am trying to imagine the future of model-based RL, which might be: model-based RL as a service (?).

Imagine you are an enterprising individual. You have some problems you want to solve;

- generating new drugs for curing cancer,
- a room that needs cleaning,
- explaining matriods,
- driving,
- poverty.

What service would help you solve them? Well, if you had an accurate, efficient model of the world, you could plan for the outcomes you want. For example, with a model of quantum physics, chemistry and human biology we could search through possible drugs for ones that will cure cancer(s).

We could have models for;
- generalised chemical reactions,
- human scale robots (indoors / outdoors). Classical physics,
- human learning,
- road rules and human behaviour,

Given a domain specification (the model of robot, the sensors, a high level description of its task)
we supply a (quickly adaptable) model of internal and external dynamics.

Question: why would anyone trust our models?

```python
request = requests.watch('https://...')
if request:
    observations, actions, reward = parse_config(request.config)
    design_model(observations, actions, reward)
```

But what about learning new domains? __Explorers__

- learns a model of a new environment
- does experiments (possibly breaks things...)

You put the explorer in your environment, and set back, well back. It quickly constructs a model of its environment.

- if allowed to mix chemicals, (should construct the periodic table, ...)
- interact with a computer (...)
- ??? ->

</div>
