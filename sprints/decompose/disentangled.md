> Carving nature at its joints

Examples of disentanglement



## Definition

Not sure independence is what we want. Take the CelebA dataset. If we constrain every latent dimension to be independent, and one of the dimensions is a smile vector, then that precludes any other dimension from altering the mouth. As the prescence of a yawn vector would explain away the smile vector. Would create a collider which makes independent variables dependent when the effect is observed.

Take Marcus' example. Faces and lighting. There are two sources of variation, the face present, an the lighting used when the image was recorded. But, lighting and faces are not independent!? No wait. They are independent, but when observed they can explain each other away!? No problem here? If we optimise the latent representation to have independent factors, it is coherent to want those factors to represent lighting/face. (???)
Obviously, if lighting=0, then I cant see anything... And while it is true that the lighting doesnt effect the face that is actually there, it does effect the image recorded. Hmm.

## Moving mnist

With 2 digits just bouncing around they are definitely independent. But what if they are allowed to bound off each other? Their appearance is independent but thei position is not.

What about having to seek out the digits on a changing scene? What about having to do arithmetic with them as well?

- want to learn some heuristics. what moves together is not independent. local relationships imply ...?

http://www.cs.toronto.edu/~nitish/unsupervised_video/
https://github.com/rszeto/moving-symbols
https://gist.github.com/tencia/afb129122a64bde3bd0c

## Resources

- [Beta VAE](https://arxiv.org/abs/1804.03599)
- [Structured disentangled representations](https://arxiv.org/abs/1804.02086)
- [ICM](https://arxiv.org/abs/1712.00961)
