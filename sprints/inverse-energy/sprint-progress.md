### 1) Motivate the idea as a solution to an existing problem

__Claim.__ A energy function will give a more efficient, succinct, robust and transferrable representation that a step function.

__Efficiency.__

- `energy -> step` It is easy to calculate steps from the energy fn. $s_{t+1} = f(s_t) = s_t - \eta \nabla E(s_t)$
- `step -> energy` It is not easy to calculate the energy with the step function. You would need to integrate. _(well, this might not matter. the computational cost of an expernsive operation that is never used is zero... Ok. this is interesting!! what do we need the energy function for? <- facilitates efficient planning. you are playing with another powerful agent. you know what they want. when planning you only bother searching for solutions that also satisfy what the other agent wants)_

<font color='red'>TODO</font> need to make this argument more formal! Comptational complexity.


__Generalisation.__

If the system being modelled is truly minimising an energy function, then using the right parameterisation will give a good inductive bias for learning, making it more efficient _(but is the gain in efficiency actually significant?)_.

<font color='red'>TODO</font> need to make this argument more formal! Sample complexity.

__Transfer.__ A communication problem between teacher and student.

Info required to communicate E is much more that dE!?

<font color='red'>TODO</font> need to make this argument more formal! Communication complexity.

__Composable.__

Using $f(s_t) = g_1(s_t) + g_2(s_t)$ will not make sense. But following the gradient of $E(s_t) = E_1(s_t) + E_2(s_t)$ will make sense.

_(hmm. i doubt this is true!? What assumptions does this actually require? How does it restrict the actual step fn?_)

### 2) Demonstrate that the "existing" problem really exists

> Why is learning an energy function hard? Is it even hard? Does it actually occur in practice?

...

### 3) Generate alternative solutions and a suitable baseline

Compare:

- vanilla neural network,
- residual neural network, (can be viewed as representing a dynamical system, $x + f(x)$)
- IEL.

### 4) Design the minimal viable experiment to falsify the proposed solution

- [X] Learn a simple 1d energy function using observations of its gradients.  (_Done. It works, but is unstable. see `simple1d.ipynb`_)
- [ ] Learn a n-d energy function from trajectories of gradient descent.
- [ ] Grid world experiment (E should learn to measure the squared distance!)
