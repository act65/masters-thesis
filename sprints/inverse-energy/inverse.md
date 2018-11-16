# Why do we care?

In my mind the inspiration for IEL comes from physics (and IRL but forget that for now). There are two valid ways to view (say) classical mechanics. As a set of causes and effects (in my mind this corresponds to a step function $x_{t+1} = f(x_t)$) or as the optimal solution of an energy functional (in this case see the [principle of least action](https://en.wikipedia.org/wiki/Principle_of_least_action)).

This idea excites me as I hope to infer what the universe is optimising given observations of the optimal steps it takes towards its goal.

Ok, enough blue skies. Lets try and imagine a few (more) concrete examples.

### Wikipedia

Let's imagine what Wikipedia might be able to do with their data. Wikipedia has millions of articles and a collective history of billions of edits. What were all of those edits attempting to achieve? They were all an attempt to 'improve' Wikipedia, but what do we mean by improve?

Let's formalise the dataset as a set of pairs $\{(s_i, e_i) : \in [1:N]\}$, where $s_i$ was current state of a document, and $e_i$ was the edit applied.

Hmm. The reward function I would write for editing Wikipedia would be something like;

- ability of the article to be easily understood (_compression_),
- completing Wikipedia with info from external sources (_completeness_),

Aka, ability to efficiently code for any/all 'important' information accessible via book, word of mouth, internet, videos, ...

Typical IRL approaches would fail at this problem.
- Part of the problem is context, and the ability to search for new information that might not be a part of Wikipedia.
- Also, the ability to be understood still only makes sense w.r.t people?

But, I imagine IRL would learn a good model of grammar...

### Robotics

One of the dreams of IRL is that we could simply expose robots to youtube videos, and they would infer the rewards people are optimising when they; collect rubbish, give CPR, ... and with these inferred rewards, they could plan and achieve the same goals.

Tbh, I am pretty skeptical of this ever working well. Maybe as a form for pretraining.

### Career advice/marketing/selling product

I can imagine a tool that predicts your reward function and uses that to match you with a career/service/product that you have been looking for (or other things you might be convinced to spend money on...).
What data would there be to infer a persons reward function?

But, how accurate can the inferred reward really be? It would require 'coverage' (like off-policy). The person must have done sufficient exploration, tried new things etc for IRL to accurately infer R.

Ok, the problem here is that we are not observing the optimal policy. Lol! We are observing a poor approximation/slowly converging solution.


### Principles of;

__Neural design.__

In the book [Principles of neural design](https://mitpress.mit.edu/books/principles-neural-design) they explore how a principle, the minimisation of energy, explains much of the brains structure, via the minimisation of wiring, computation with chemistry, ...

The principle of minimising wiring can be easily described, yet it has extrodinary power to generate predictions. Given the right initial conditions, this principle can explain wiring in the cortex, cerebellum, and even microchips!

__Entropy.__

We know that entropy always increases. Ie the relationship of the energy function between two states $E(s_{t}) \le E(s_{t+1})$. But knowing how, exactally, the current state transitions to the next state contains far more info/complexity.

__Colloids.__

If we remove all the physics from a simulation of colloids, but maximise each colloids position based on its (local) entropy (so that each colloid has the most freedom/space). The crystalline structures emergy from the interaction of a system of colloids. So the shape of the colloids and the principle of entropy maximisation is enough to explain many different crystalline strucutres.

### Inverse market design

If we looked at the allocation of wealth and power to individuals, what can we infer about the allocation mechanism doing the allocation? We could assume that the current allocation is optimal under some measure of utility, and attempt to infer that utility function.
