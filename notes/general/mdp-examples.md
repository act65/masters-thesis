- __States:__
- __Actions:__
- __Transition fn:__
- __Reward fn:__

http://www.it.uu.se/edu/course/homepage/aism/st11/MDPApplications1.pdf
https://www.worldscientific.com/worldscibooks/10.1142/p809

> If they were not intercorrelated, one would simply minimize expected loss in a greedy fashion, starting with catnip etc; but as they are intercorrelated, now a drug has both direct value (if the cat responds) and value of information (its failure gives evidence about what other drugs that cat might respond to), which means the greedy policy may no longer be the optimal policy.
https://www.gwern.net/Catnip#optimal-catnip-alternative-selection-solving-the-mdp

The general feeling of an MDP.
- Actions need to be adapted to new observations and contexts.
- While instantaneous results are good, we care about the longer term aggregates.

#### Bus engine replacement

- __States:__ Accumulated mileage of each bus (since their last replacement).
- __Actions:__ Replace bus $i$, Y / N. If Y, what year model to replace with?
- __Transition fn:__ How the mileage of each bus changes between fitness checks.
- __Reward fn:__ Age dependent recurring cost - for repairs - and replacement cost.

Note: The transition function could be diagonal, or not. If diagonal then buses are not able to effect the milage of other busses, possibly by taking another's shift. In this case, the problem is reduced to a contextual bandit problem (?).

[Putterman]()

#### The ALOHA protocol

- __States:__ For each terminal, is last attempt a collision.
- __Actions:__ The probability of each terminal attempting to send a new packet (if there has been a collision).
- __Transition fn:__ Combines terminal packets into either a successful transmission, or a collision.
- __Reward fn:__ If a packet is sent, great. If not, bad.

[Putterman]()

#### Mate desertion in Cooper's Hawks

- __States:__ The product of the brood's health and mother's health ($[2:7] \times [2:7]$).
- __Actions:__ Stay, hunt, desert.
- __Transition fn:__ The four developmental stages, early nestling, late netling, early feldgling, late feldgling. From one developmental stage to the next, the energy levels of the mother and brood are determined by the initial energy reserves, the actions taken and the availability of food. (this was estimated from data data gatherd by ...)
- __Reward fn:__

[Putterman]()

#### Which cubicle should I use?

- __States:__ The occupancy of the $N$ cubicles.
- __Actions:__ Choose a cubicle. (cannot move half way through)
- __Transition fn:__ A new person might join the ???, and someone might exit.
- __Reward fn:__ Outer cubiles are the most used, so avoiding the is better. But sitting next to someone is also bad.

Not a sequential decision problem? But has long term payoffs??

#### Managing graduate study

- __States:__ Product of, progress x time left x motivation x confusion.
- __Actions:__ Quit, read, write, experiment,
- __Transition fn:__ Writing increases progress a small amount, experimentation may increase progress, ...
- __Reward fn:__ Sanity.

#### Best papers

- __States:__ ['X gpus', 'deep', 'p-value', 'Michael Jordan', '', ]
- __Actions:__ Should I continue reading this paper? Y/N.
- __Transition fn:__
- __Reward fn:__ Papers are classified as good versus bad. Want to spend max time reading good papers.


- An animal stockpiling food?!
-
