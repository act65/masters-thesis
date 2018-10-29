Imagine you are given two models $f, g$ that might predict pedestrian and traffic behaviour respectively. How can you safely/sensibly combine their predictions? If the models were provided as densities we could evaluate the next step as $s_{t+1} = \mathop{\text{argmax}}_{s_{t+1}} \prod_i p_i(s_{t+1} \mid s_t)$.


### Motivation

Existing problem is?

### Minimal viable experiment

Learn n different experts on the mnist digits, each learning a single number.
Attempt to combine the n experts.
