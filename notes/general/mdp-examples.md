Examples of MDPs in the wild.

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

[Putterman pg 8]()

#### Mate desertion in Cooper's Hawks

- __States:__ The product of the brood's health and mother's health ($[2:7] \times [2:7]$).
- __Actions:__ Stay, hunt, desert.
- __Transition fn:__ The four developmental stages, early nestling, late netling, early feldgling, late feldgling. From one developmental stage to the next, the energy levels of the mother and brood are determined by the initial energy reserves, the actions taken and the availability of food. (this was estimated from data data gatherd by ...)
- __Reward fn:__

[Putterman pg 10]()

#### But who's counting?

- __States:__ A random number, and the value of each of five possible locations. Possibly none value.
- __Actions:__ Choose which location to add the latest random number.
- __Transition fn:__ Deterministically updates the storage location given the action and observed random number.
- __Reward fn:__ The total magnitude of the stored number, only given after the storage is full..

[Putterman pg 13]()

#### Diagnosing catnip immunity

- __States:__ The truth values for immuity to an of the 4 drugs ( Catnip / Valerian / Silvervine / Honeysuckle )
- __Actions:__ Choose which drug to test.
- __Transition fn:__ Updates the truth values with some probability of returning a true postive or false negative.
- __Reward fn:__ Minimize the cost to find a working drug. Catnip=$8.96, Valerian=$7.00, Silverine=$17.77, Honeysuckle=$7.99.

> Bol et al 2017, as noted, provides responses for 4 drugs (catnip/Valerian/silvervine/honeysuckle) in a large sample of cats; responses turn out to be heavily intercorrelated, permitting the ability to better predict responses to the catnip alternatives based on a known response to one of the others. This becomes useful if we treat it as a drug selection problem where we would like to find at least one working drug for a cat while saving money, and adapting our next test based on failed previous tests.

> If they were not intercorrelated, one would simply minimize expected loss in a greedy fashion, starting with catnip etc; but as they are intercorrelated, now a drug has both direct value (if the cat responds) and value of information (its failure gives evidence about what other drugs that cat might respond to), which means the greedy policy may no longer be the optimal policy.

[gwern on Catnip](https://www.gwern.net/Catnip#optimal-catnip-alternative-selection-solving-the-mdp)

This one is interesting. The four actions effect only the four state element-wise. But our knowledge that certain immunities are correlated make it possible to intelligently guess which tests should be performed.

<!-- #### Which cubicle should I use?

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

- __States:__ Who you have dated in the past, and how it ended (good vs bad).
- __Actions:__ Choose a new person to date.
- __Transition fn:__
- __Reward fn:__ ??? -->


#### Ad targeting



#### Youtube recommendation




#### Salamon harvesting

- __States:__ The size of the salamon population.
- __Actions:__ The size of the salamon population  to be left to spawn.
- __Transition fn:__ Given the number left to spawn, it returns the size of the salamon population in the next season.
- __Reward fn:__ The size of salamon population harvested.

YOu might call this MDP a one dimensional MDP, as there is only a single dimension that is acted up, is transitioned, is rewarded... More salamon

[Real applications of MDPs](http://www.it.uu.se/edu/course/homepage/aism/st11/MDPApplications1.pdf)

#### Fire engine allocation

- __States:__ The magnitude of a fire. The type of alarm. And the total number of first and second fire engines already deployed.
- __Actions:__ Whether to send more fire engines.
- __Transition fn:__ Given the number of fire engines fighting the fire, and the fires type / magnitude, the building may be destroyed or saved. Fire may start at anytime, in a random location throughout the city.
- __Reward fn:__ Damage incurred by the fires.

[Real applications of MDPs](http://www.it.uu.se/edu/course/homepage/aism/st11/MDPApplications1.pdf)


Other possible MDPs?

- An animal stockpiling food?!
- Robotics / movement


***

More Refs

- http://www.it.uu.se/edu/course/homepage/aism/st11/MDPApplications1.pdf
- https://www.worldscientific.com/worldscibooks/10.1142/p809



\section{MDP examples}\label{MDP-examples}

Here are some examples of MDPs in the wild.

\paragraph{Bus engine replacement}

\begin{itemize}
\tightlist
\item
  \textbf{States:} Accumulated mileage of each bus (since their last
  replacement).
\item
  \textbf{Actions:} Replace bus \(i\), Y / N. If Y, what year model to
  replace with?
\item
  \textbf{Transition fn:} How the mileage of each bus changes between
  fitness checks.
\item
  \textbf{Reward fn:} Age dependent recurring cost - for repairs - and
  replacement cost.
\end{itemize}

Note: The transition function could be diagonal, or not. If diagonal
then buses are not able to effect the mileage of other buses (a bus can effect the mileage of another bus by taking its shift).
In the diagonal case, the problem is reduced to a contextual bandit problem.

\cite{Putterman2015}

\hypertarget{the-aloha-protocol}{%
\paragraph{The ALOHA protocol}\label{the-aloha-protocol}}

\begin{itemize}
\tightlist
\item
  \textbf{States:} For each sender, was the last attempt to send a collision.
\item
  \textbf{Actions:} The probability of each sender attempting to send
  a new packet.
\item
  \textbf{Transition fn:} Combines sender's packets into either a
  successful transmission, or a collision.
\item
  \textbf{Reward fn:} If a packet is sent, great. If not, bad.
\end{itemize}

\cite{Putterman2015} pg 8

\hypertarget{mate-desertion-in-coopers-hawks}{%
\paragraph{Mate desertion in Cooper's
Hawks}\label{mate-desertion-in-coopers-hawks}}

\begin{itemize}
\tightlist
\item
  \textbf{States:} The product of the brood's health and mother's health
  (\([2:7] \times [2:7]\)).
\item
  \textbf{Actions:} Stay, hunt, desert.
\item
  \textbf{Transition fn:} The four developmental stages, early nestling,
  late nestling, early fledgling, late fledgling. From one developmental
  stage to the next, the energy levels of the mother and brood are
  determined by the initial energy reserves, the actions taken and the
  availability of food. (this was estimated from data gathered by
  \ldots{})
\item
  \textbf{Reward fn:}
\end{itemize}

\cite{Putterman2015} pg 10

\hypertarget{but-whos-counting}{%
\paragraph{But who's counting?}\label{but-whos-counting}}

\begin{itemize}
\tightlist
\item
  \textbf{States:} A random number, and the value of each of five
  possible locations. Possibly none value.
\item
  \textbf{Actions:} Choose which location to add the latest random
  number.
\item
  \textbf{Transition fn:} Deterministically updates the storage location
  given the action and observed random number.
\item
  \textbf{Reward fn:} The total magnitude of the stored number, only
  given after the storage is full..
\end{itemize}

\cite{Putterman2015} pg 13

\hypertarget{diagnosing-catnip-immunity}{%
\paragraph{Diagnosing catnip
immunity}\label{diagnosing-catnip-immunity}}

\begin{itemize}
\tightlist
\item
  \textbf{States:} The truth values for immunity to an of the 4 drugs (
  Catnip / Valerian / Silvervine / Honeysuckle )
\item
  \textbf{Actions:} Choose which drug to test.
\item
  \textbf{Transition fn:} Updates the truth values with some probability
  of returning a true positive or false negative.
\item
  \textbf{Reward fn:} Minimize the cost to find a working drug.
  Catnip=\$8.96, Valerian=\$7.00, Silverine=\$17.77, Honeysuckle=\$7.99.
\end{itemize}

\begin{quote}
Bol et al 2017, as noted, provides responses for 4 drugs
(catnip/Valerian/silvervine/honeysuckle) in a large sample of cats;
responses turn out to be heavily intercorrelated, permitting the ability
to better predict responses to the catnip alternatives based on a known
response to one of the others. This becomes useful if we treat it as a
drug selection problem where we would like to find at least one working
drug for a cat while saving money, and adapting our next test based on
failed previous tests.
\end{quote}

\begin{quote}
If they were not intercorrelated, one would simply minimize expected
loss in a greedy fashion, starting with catnip etc; but as they are
intercorrelated, now a drug has both direct value (if the cat responds)
and value of information (its failure gives evidence about what other
drugs that cat might respond to), which means the greedy policy may no
longer be the optimal policy.
\end{quote}

\href{https://www.gwern.net/Catnip\#optimal-catnip-alternative-selection-solving-the-mdp}{gwern
on Catnip}

This one is interesting. The four actions effect only the four state
element-wise. But our knowledge that certain immunities are correlated
make it possible to intelligently guess which tests should be performed.

\hypertarget{ad-targeting}{%
\paragraph{Ad targeting}\label{ad-targeting}}

\hypertarget{youtube-recommendation}{%
\paragraph{Youtube recommendation}\label{youtube-recommendation}}

\hypertarget{salamon-harvesting}{%
\paragraph{Salamon harvesting}\label{salamon-harvesting}}

\begin{itemize}
\tightlist
\item
  \textbf{States:} The size of the salmon population.
\item
  \textbf{Actions:} The size of the salmon population to be left to
  spawn.
\item
  \textbf{Transition fn:} Given the number left to spawn, it returns the
  size of the salmon population in the next season.
\item
  \textbf{Reward fn:} The size of salmon population harvested.
\end{itemize}

You might call this MDP a one dimensional MDP, as there is only a single
dimension that is acted up, is transitioned, is rewarded\ldots{} More
salmon

\href{http://www.it.uu.se/edu/course/homepage/aism/st11/MDPApplications1.pdf}{Real
applications of MDPs}

\hypertarget{fire-engine-allocation}{%
\paragraph{Fire engine allocation}\label{fire-engine-allocation}}

\begin{itemize}
\tightlist
\item
  \textbf{States:} The magnitude of a fire. The type of alarm. And the
  total number of first and second fire engines already deployed.
\item
  \textbf{Actions:} Whether to send more fire engines.
\item
  \textbf{Transition fn:} Given the number of fire engines fighting the
  fire, and the fires type / magnitude, the building may be destroyed or
  saved. Fire may start at any time, in a random location throughout the
  city.
\item
  \textbf{Reward fn:} Damage incurred by the fires.
\end{itemize}

\href{http://www.it.uu.se/edu/course/homepage/aism/st11/MDPApplications1.pdf}{Real
applications of MDPs}

Other possible MDPs?

\begin{itemize}
\tightlist
\item
  An animal stockpiling food?!
\item
  Robotics / movement
\end{itemize}

\begin{center}\rule{0.5\linewidth}{\linethickness}\end{center}

More Refs

\begin{itemize}
\tightlist
\item
  http://www.it.uu.se/edu/course/homepage/aism/st11/MDPApplications1.pdf
\item
  https://www.worldscientific.com/worldscibooks/10.1142/p809
\end{itemize}
