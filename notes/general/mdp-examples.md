- __States:__
- __Actions:__
- __Transition fn:__
- __Reward fn:__

http://www.it.uu.se/edu/course/homepage/aism/st11/MDPApplications1.pdf
https://www.worldscientific.com/worldscibooks/10.1142/p809

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


#### Diagnosing catnip immunity

- __States:__ The truth values for immuity to an of the 4 drugs ( Catnip / Valerian / Silvervine / Honeysuckle )
- __Actions:__ Choose which drug to test.
- __Transition fn:__ Updates the truth values with some probability of returning a true postive or false negative.
- __Reward fn:__ Minimize the cost to find a working drug. Catnip=$8.96, Valerian=$7.00, Silverine=$17.77, Honeysuckle=$7.99. 

> Bol et al 2017, as noted, provides responses for 4 drugs (catnip/Valerian/silvervine/honeysuckle) in a large sample of cats; responses turn out to be heavily intercorrelated, permitting the ability to better predict responses to the catnip alternatives based on a known response to one of the others. This becomes useful if we treat it as a drug selection problem where we would like to find at least one working drug for a cat while saving money, and adapting our next test based on failed previous tests.

> If they were not intercorrelated, one would simply minimize expected loss in a greedy fashion, starting with catnip etc; but as they are intercorrelated, now a drug has both direct value (if the cat responds) and value of information (its failure gives evidence about what other drugs that cat might respond to), which means the greedy policy may no longer be the optimal policy.

[https://www.gwern.net/Catnip#optimal-catnip-alternative-selection-solving-the-mdp](https://www.gwern.net/Catnip#optimal-catnip-alternative-selection-solving-the-mdp)

This one is interesting. The four actions effect only the four state element-wise. But our knowledge that certain immunities are correlated make this ...?!?

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


#### Dating

Dating. In the past you dated then/their friends. Now it effects your chances.





- An animal stockpiling food?!
-
