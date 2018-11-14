# Why do we care?

> What is the future of inverse learning?

Inverse learning can be summarised as learning a functional that is being optimised.

Classical mechanics can be modelled as minimising the 'action'.
People and learned agents can be modelled as maximising their rewards.
X can be modelled as ???.  

The future of marketing: facebook (/similar) will observe you and infer the experiences you find rewarding.

In nature, we commonly observe the optimal policy under the [principle of least action](https://en.wikipedia.org/wiki/Principle_of_least_action). The lagrangian defines an energy function, say of ... which is minimised.

In economies (or social environments) there are many agents with their own agendas. We tend to take actions that maximise our own rewards.




Big questions

- Is it a universal approximator? (what can we not do with it?)
- In which cases does it work (better)?
- Global view versus local view

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

But I imagine IRL would learn a good model of grammar.

### Robotics

Expert demonstrations of CPR. Collecting rubbish.

I am actually pretty skeptical of IRL's utility here.

The ability to learn from Youtube. What a nightmare.


### Career advice/marketing/selling product

IRL that predicts your reward function and uses that to match you with a career/service/product.
What data would there be to infer a persons reward function. Well if you are facebook, then pretty much all of it.

But, how accurate can the inferred reward really be? It would require 'coverage' (like off-policy). The person must have done sufficient exploration, tried new things etc.

Ok, the problem here is that we are not observing the optimal policy. Lol! We are observing a poor approximation/slowly converging solution.

### Communication

When talking to friends, we tend to say things like, ???.


Is there a notion is psych or ling of goalbased communication?


### Principles of;

Neural design.

In the book [Principles of neural design](https://mitpress.mit.edu/books/principles-neural-design) they show explore how a principle, the minimisation of energy, explains much of the brains structure, via the minimisation of wiring, computation with chemistry, ...

The principle of minimising wiring can be easily described, yet it has extrodinary power to generate predictions. Given the right initial conditions, this principle can explain wiring in the cortex, cerebellum, and even microchips!

Entropy

We know that entropy always increases. Ie the relationship of the energy function between two states $E(s_{t}) \le E(s_{t+1})$. But knowing how exactally the current state transitions to the next state contains far more info/complexity.


### Inverse market design

The allocation of scarce resources.

If we looked at the allocation of wealth and power to individuals, what can we infer about the allocation (/market) machanism doing the allocation?


## Learning inversions

Hmph. What about getting access to non-optimal trajectories. It seems that this could increase the speed of learning (if the non-optimal trajectories were chosen correctly). We receive pairs indicating the dicision boundaries, good-bad.

We dont get access to any negative samples. How does this restrict out ability to learn and generalise!?
