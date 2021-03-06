A timeline of my masters.

1. In my proposal I stated that I wanted to understand __transfer__ and __HRL__. With the goal of understanding and / or improving an agents ability to learn.
2. I completed four 'sprints'; __HRL, exploration, IRL, disentanglement__, each of two weeks. After these sprints I decided to focus on __action abstractions__, which can be related to disentanglement (see [independent actions](http://willwhitney.com/assets/papers/Disentangling.video.with.independent.prediction.pdf)) and seems to be required for meaningful transfer.
4. After an exploration of 'action abstractions', I started to understand that HRL is a special case of __abstraction__; one of a special heirarchical structure, and which is focused on achieving a temporal abstraction. I also became aware that no abstraction is likely to do better or worse than any other, in the general case (once you account for the complexity of the abstraction -- this is a result of the no-free-lunch theorem). Also, I became less enthused with action abstractions as there were some rather straight forward experiments that could be done, but I couldn't see how they might help me understand when / why action abstractions may or may not help.
5. I explored more general results about abstractions. I was especially interested in theoretical ways to analyse __[Near optimal abstractions](https://arxiv.org/abs/1701.04113)__. I hoped this would help me understand; how much reward can you expect to get given that you are using an abstraction with certain properties.
6. I spent some time thinking about why we care about abstractions. We want to throw away the unimportant parts, so we can focus on the essential. The point is to make the problem easier to solve, in some sense. This led to __Solvable abstractions__. How can we find an abstraction that is easily solvable?
7. After reading a few papers on the theory of RL, I decided I wanted a better understanding of __MDPs__, which were the main setting considered in the proofs I had been attempting to understand.
8. I found a great paper (__[Value function polytope](https://arxiv.org/abs/1901.11524)__) that gave insight into the structure of the MDP and it's optimisation. I explored this further and combined it with some theoretical work on [acceleration via overparameterisation](https://arxiv.org/abs/1802.06509).
9. Now, with my new understanding of MDPs, I returned to the problem of abstraction. A simple, and easily solvable system is a linear one. Is there a way find and exploit linearity within an MDP? __[Linear MDPs](https://www.pnas.org/content/106/28/11478)__? But what do we mean by linear? ...
10.



\begin{enumerate}
  \tightlist
  \item In my proposal I stated that I wanted to understand \textit{transfer} and \textit{HRL}. With the goal of understanding and / or improving an agents ability to learn.
  \item I completed four 'sprints'; \textit{HRL, exploration, IRL, disentanglement}, each of two weeks. After these sprints I decided to focus on \textit{action abstractions},
  which can be related to disentanglement.
  \item I explored more general results about abstractions. I was especially interested in theoretical ways to analyse MDPs such as near optimality \cite{Abel2017}.
  \item Something about action abstractions.
  \item I understood that HRL is a special case of abstraction; one of a special hierarchical structure, and which is focused on achieving a temporal abstraction. I also became aware that no abstraction is likely to do better or worse than any other, in the general case (once you account for the complexity of building the abstraction -- this is a result of the no-free-lunch theorem).
  \item I went to a conference and met Theja Tulabandhula. He was presenting his paper on symmetry based abstractions for RL \cite{Mahajan2017}. This is exactly what I had been looking for, but didn't know it. I promised myself I would look into it further.
  \item After reading a few papers on the theory of RL, I decided I wanted a better understanding of MDPs, which were the main setting considered in the proofs I had been attempting to understand.
  \item I found a great paper, the Value function polytope \cite{Dadashi2018}, that gave insight into the structure of the MDP and it's optimisation. I explored this further and got curious about how to understand optimisation, overparameterisation and dynamics for RL.
  \item With my new understanding of MDPs, I returned to the problem of abstraction. A simple, and easily solvable system is a linear one. Is there a way find and exploit linearity within an MDP?
\end{enumerate}
