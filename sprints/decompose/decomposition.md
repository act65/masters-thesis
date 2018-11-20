An alternative view on disentanglement. But what is its relationship to independence?

***

We want n different modules that specialise in their tasks? How can we learn different specialists?

- winner takes all credit assignment. positive feedback/lateral inhibition/...?
- give access to different inputs/resources. they physically/computationally cannot do the same thing...
- train on different tasks...
- ?


***

Ok, imagine we give two specialists different inputs (say, left-right halves of mnist) and train them separately on the same classification task. Their outputs are __not__ going to be independent... The fact the the left half classifies its input as a 2, makes it likely that the right half will also classify its input as a 2.

So, in this instance, the two modules will output independent results if;

- they are trained on the same task yet receive independent inputs
- the two experts are trained on different tasks, which are independent
- ?

***


Questions
- __Specialisation__! Independently useful contributions.
- __Q__ How is independence related to decomposition?

## Resources

- [MoE](https://arxiv.org/abs/1701.06538) <-- could measure the hidden states MI/TC!?
- ?
