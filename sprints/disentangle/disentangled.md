> Carving nature at its joints

Examples of disentanglement

## Definition

What do we mean by disentangled?

- A popular definition seems to be statistical independence.
- The latent variable to match the generating variable types. (the ability to construct sets equipped with a metric/product/transform!? mnist -> a dimension with 10 categorical variables, a ring of reals describing azimuth, bounded dims describing translation, ...)
-

## Moving mnist

With 2 digits just bouncing around they are definitely independent. But what if they are allowed to bounce off each other? Their appearance is independent but their position is now conditional.

What about having to seek out the digits on a changing scene? What about having to do arithmetic with them as well?

- want to learn some heuristics. what moves together is not independent. local relationships imply ...?

http://www.cs.toronto.edu/~nitish/unsupervised_video/
https://github.com/rszeto/moving-symbols
https://gist.github.com/tencia/afb129122a64bde3bd0c

## Representation

What if I have an independent variable that is a ring, or has some other topology other than a line?

## Resources

- [Beta VAE](https://arxiv.org/abs/1804.03599)
- [Structured disentangled representations](https://arxiv.org/abs/1804.02086)
- [ICM](https://arxiv.org/abs/1712.00961)
- [Information dropout](https://arxiv.org/abs/1611.01353)
