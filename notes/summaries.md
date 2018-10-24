## World models

- Learn a large RNN that approximates the transition function.
- Add noise when simulating to avoid the controller overfitting to bias in the model.
- Can use the world-model to imagine training experience for training the controller.

https://worldmodels.github.io/

## Temporal difference models

- Define a proxy reward as the distance between a goal state and the current state.
- Learn a Q value that estimates the reachability of different states within t steps.
- It turns out when t>>1 we recover model-free learning and yet we can still plan with t~=1.

https://bair.berkeley.edu/blog/2018/04/26/tdm/

## Learning to reinforcement learn

- Use a RNN as an actor-critic but also provide the reward recieved as an input.
- Train weights over sampled MPDs to maximise cumlative reward over an episode.
- Freeze weights and test on new MPDs.

https://arxiv.org/abs/1611.05763
