AlphaGo can use its knowledge of the rules to simulate possible futures.
It can evaluate an action taken by exploring where that action is likely to lead, to a win or a loss. This 'expensive oracle' can be used to provide supervision for a learned policy.

I can imagine a 'memorizer' that stores past state-action values. For every state-action ever seen, it memorizes the discounted return recieved after finishing the episode. This could be used as an 'expensive oracle' to provide supervision for a learned policy.

In contrast to AlphaGo's expensive oracle, which is model-based, this oracle is model-free.


Ok. Why doesn't this work?

- Use state-actions to look up a value.
- Use states to look up action values.


We are trying to approximate a $Q$-function with memory. Where $Q(s, a) = V[NN(s)]$


$$
Q(s, a) = V[NN(s;a)], \;\; V \in R^N\\
Q(s, a) = V[NN(s)][a], \;\;V \in R^{N \times |A|} \\
$$


$$
armax {[\sigma (d[NN(s)]) \cdot \sigma (v[NN(s)])]}_{i}
$$


Many neighbors, few neighbors.
Want to generalise...


Not going to be sample efficient.
But what do we gain?
Stability?
One-shot learning?
Reduction to supervised learning.


- [Organizing experience](https://arxiv.org/abs/1806.04624)
- [Striving for simplicity in off-policy DRL](https://arxiv.org/abs/1907.04543)
- [Large memory layers with product keys](https://arxiv.org/abs/1907.05242)