## Solveable representations

> Representations with structure that is easily solvable.

While there are other ways to add exploitable structure, here we only consider linearity.

The bellman equation is a non-linear optimisation problem. It does have some nice properties, like having a unique optima under the bellman operator. But, in general, it isn't very friendly. Is there a way to turn this into a linear problem? What sacrifices need to be made to achieve this?

### Why linearity?

- it has many mathematical tools for analysis.
- we know linear systems can be solved efficiently.
- ?

Linearity is a nice property that makes optimisation simpler and more efficient.

- Linear programming (see appendix: LP)
- Linear markov decision processes

Solving a system of linear relationships. Has a complexity of ???.

In fact. MDPs can actually be solved via LP. see [appendix].

### A closer look at LMDPs

(the Todorov ones...)

The three steps of abstraction
- relaxation (transform to a new domain)
- linearisation (and solve)
-


#### LMDPs; more formally

![''](../../../pictures/drawings/abstract-representations-solve.png){ width=450px }

Pick $a \in A$, versus, pick $\Delta(S)$. $f: S\to A$ vs $f:S \to \Delta(S)$.

In the original Todorov paper, they derive the LMDP equations for minimising a cost function. This maximisation derivation just changes a few negative signs around. Although there is also a change in the interpretation of what the unconstrained dynamics are doing. ...?

$$
\begin{align}
V(s) &= \mathop{\text{max}}_{u} q(s) - \text{KL}(u(\cdot| s) \parallel p(\cdot | s)) + \gamma \mathop{\mathbb E}_{s' \sim u(\cdot | s)} V(s') \tag{1}\\
\\
u^{* }(\cdot | s) &= \frac{p(\cdot | s)\cdot z(\cdot)^{\gamma}}{\sum_{s'} p(s' | s) z(s')^{\gamma}} \tag{8}\\
z_{u^{* }} &= e^{q(s)}\cdot P z_{u^{* }}^{\gamma} \tag{11}\\
\end{align}
$$

By definition, an LMDP is the optimisation problem in (1). (3) Define a new variable, $z(s) = e^{v(s)}$. (5) Define a new variable that will be used to normalise $p(s' | s)z(s')^{\gamma}$. (8) Set the optimal policy to minimise the KL distance term. (9) Since we picked the optimal control to be the form in (8), the KL divergence term is zero. (11) Rewrite the equations for the tabular setting, giving a $z$ vector, uncontrolled dynamics matrix.

(see appendix [] for a full derivation)

#### A relaxed MDP

![''](../../../pictures/drawings/abstract-representations-linear.png){ width=450px }

> Ok great, we can solve LMDPs. But how does being able to solve an LMDP help us solve MDPs?

We want a way to transform a MDP into a LMDP, while preserving the 'structure' of the MDP. But what do we mean by a MDP's structure?

The LMDP, $\{S, p, q, \gamma\}$ should;

- be able to represent the same transition dynamics as the original MDP,
- give the the same rewards was the original MDP,
- have the same optima.

(It turns out that (1) and (2) imply (3) given some assumptions. See [Optimality]())

So, given a reward function, $r$, and a transition function, $P$, from the MDP, we must translate them into a $p$ and a $q$. Thus we have built a LMDP with the same 'structure'.


\begin{aligned}
\forall s, s' \in S, \forall a \in A, \exists u_a& \;\;\text{such that;} \\
P(s' | s, a) &= u_a(s'|s)p(s'|s) \tag{1}\\
r(s, a) &= q(s) - \text{KL}(P(\cdot | s, a) \parallel u_a(\cdot| s) ) \tag{2}\\
\end{aligned}


Which leads to $|A|$ linear equations to solve, for each state in the MDP.



See appendix [] for more details.


Alternative views of linearisation.

- A relaxation of the MDP
- Linelihood interpretation

##### Unconstrained dynamics and state rewards

> Let's try and understand this thing we have contructed.

The state rewards are not capable of giving rewards for actions taken. Rather, the differences in reward, by taking another action, is captured by the KL divergence between the control and the unconstrained dynamics.

- What is their function?
- What do they look like?

Does it make sense to treat the q(s) like rewards?!
They reward for bing in state s.
But cant capture action specific rewards!?

#### Decoding

![''](../../../pictures/drawings/abstract-representations-project.png){ width=450px }

Ok, so now we get a glimpse at why LMDPs are an interesting abstraction.
THe LMDP has disentangled the search for the behaviour (go to this or that state) and the search for optimal controls (how to actually achieve that behaviour). This can be seen in the decoding step. As we know which states we want to be in, via the optimal control from solving the LMDP, $u^{* }$, but, we do not know how to implement those controls using the actions we have available.

> Two 'simpler' problems. Easier to solve?

<!--This is where most of the complexity of the RL problem is?!?-->

\begin{aligned}
P_{\pi}(\cdot | s) = \sum_a P(\cdot | s, a) \pi(a | s) \\
\pi = \mathop{\text{argmin}}_{\pi} \text{KL}\Big(u(\cdot | s))\parallel P_{\pi}(\cdot | s)\Big)
\end{aligned}


Maybe this isnt enough? Do we need to add a reward sensitive part as well?!?
(but what if the actual path we take to get there has a neg rewards?!?)

#### Optimality of solutions via LMDPs

> Do these two paths lead to the same place?
<!-- insert quote?! -->

One of the main questions we have not addressed yet is; if we solve the MDP directly, or linearise, solve and project, do we end up in the same place? This is a question about the completeness of our abstraction. Can our abstraction represent (and find) the same solutions that the original can?


\begin{aligned}
\parallel V_{\pi^{* }} - V_{\pi_{u^{* }}} \parallel_{\infty}&= \epsilon  \tag{1}\\
&=\parallel (I - \gamma P_{\pi^{* }})^{-1}r_{\pi^{* }} - (I - \gamma P_{\pi_{u^{* }}})^{-1}r_{\pi_{u^{* } }} \parallel_{\infty} \tag{2}\\
&\le\parallel (I - \gamma P_{\pi^{* }})^{-1}r - (I - \gamma P_{\pi_{u^{* }}})^{-1}r \parallel_{\infty} \tag{3}\\
&=\parallel \bigg((I - \gamma P_{\pi^{* }})^{-1} - (I - \gamma P_{\pi_{u^{* }}})^{-1} \bigg) r \parallel_{\infty} \tag{4}\\
&\le r_{\text{max}} \parallel (I - \gamma P_{\pi^{* }})^{-1} - (I - \gamma P_{\pi_{u^{* }}})^{-1}   \parallel_{\infty} \tag{5}\\
&= r_{\text{max}} \parallel \sum_{t=0}^{\infty} \gamma^t P_{\pi^{* }} - \sum_{t=0}^{\infty} \gamma^t P_{\pi_{u^{* }}}  \parallel_{\infty} \tag{6}\\
&= r_{\text{max}} \parallel \sum_{t=0}^{\infty} \gamma^t (P_{\pi^{* }} - P_{\pi_{u^{* }}})   \parallel_{\infty} \tag{7}\\
&= \frac{r_{\text{max}}}{1-\gamma} \parallel P_{\pi^{* }} - P_{\pi_{u^{* }}} \parallel_{\infty} \tag{7}\\
\end{aligned}


(1) We want to compare the optimal policies value and the value achieved by the optimal LDMP solution.
(2) Assume that there exists a policy that can generate the optimal control dynamics (as given by the LMDP). In that case we can set $P_{\pi_{u^{* }}} = U^{* }$.
(3) $r_{u^{* }}$ doesnt really make sense as the reward is action dependent. We could calculate it as $r_{\pi_{u^{* } }}$, but we dont explicity know $\pi_{u^{* }}$. $(I - \gamma P_{\pi^{* }})^{-1}r$ represents the action-values, or $Q$ values. By doing this exhange, we might over estimate the diffference under the infinity norm as two non-optimal actions may have larger difference. Also, use the element wise infinity norm.

***

Ok, great.
Insights from optimality bounds.


Need to be able to approximate the optimal controls.
When is it hard to approximate the optimal controls?
When our basis set of distributions oer future states (aka our actions) have little weight...?

Potential solution?
Use options.

#### Option decoding

What about using options to help solve the optimal control decoding?
Does this actually help?!

\begin{aligned}
P_{\pi}(\cdot | s) = \sum_\omega P_k(\cdot | s, \omega) \pi(\omega | s) \\
\pi = \mathop{\text{argmin}}_{\pi} \text{KL}\Big(u(\cdot | s))\parallel P_{\pi}(\cdot | s)\Big)
\end{aligned}

Options would allow greater flexibility in the $P_{\pi}(\cdot | s)$ distribution, making is possible to match $u(s'|s)$ with greater accuracy (and possibly cost).

- First need to demonstrate that action decoding is lossy.
- Then show that using options is less lossy.

This introduces dangers?!? As an option might accumulate unknown rewards along the way!??

### The complexity of solutions via LMDPs

> Is my path actually shorter?

The whole point of this abstraction was to make the problem easier to solve. So hasit actually made it any easier?

The complexity of solving our abstraction can be broken down into the three steps;

- linearisation:  $|S| \times \text{min}(|S|,|A|)^{2.3}$
- solve the LMDP: $\text{min}(|S|,|A|)^{2.3}$
- project back: $???$

Giving a total complexity of ...

Contrasted with the complexity of solving an MDP.

### Scaling to more complex problems

Now that we have some evidence that this LMDP solution strategy makes sense, it efficiently (see [complexity]()) yields high value (see [optimality]()) policies.
We want to test it out on some real world problems.
But the real world isn't as nice as the setting we have been working in. There are a few added complexities;

- sample based / incremental
- large / cts state spaces
- sparse rewards
<!-- more?!? -->

So now that we have explored LMDPs, how can we extract their nice properties into an architecture that might scale to more complex problems: larger state spaces and action spaces, sparse rewards, ...?


#### Incremental implementation

Generalise to a more complex problem. We are only given samples.
A first step to tackling more complex problems.



##### Model based
Learn $p, q$ based on samples.

\begin{aligned}
\mathcal L(\theta, \phi) = \mathop{\mathbb E}_{s, a,} \bigg[ r(s, a) - q_\theta(s) + \text{KL}(p_\phi(\cdot | s) \parallel P(\cdot | s, a)) \bigg]\\
\mathcal L(\theta, \phi) = \mathop{\mathbb E}_{s, r, s'} \bigg[r - q_\theta(s) - p_\phi(s' | s) \log \frac{1}{ p_\phi(s' | s)} \bigg] \\
\end{aligned}



***

Ok. Lets take a different approach. __Q:__ Why is it a bad idea to try to do incremental RL with this linearisation trick?
Not sure.

***

Alternative perspective. The high value trajectories are the most likely ones.

### Distributions over states

What if we wanted to approximate these distributions?
Generalise subgoal methods to work with distributions?
The distribution could be constructed via; parzen window / GMM, neural flow, ?!.

Connections to distributional RL?

Questions

- What is p(s'|s)!?!?
- Want some examples of MDPs they cannot solve.
- What is the relationship to other action embedding strategies?
- How does p(s'|s) bias the controls found??? I can imagine the unconstrained dynamics acting as a prior and prefering some controls over others.
- If we have m states and n actions. Where m >> n. Then $u(s'|s)$ is much larger than $\pi(a|s)$. Also, $u(s'|s)$ should be low rank?! $u_{s's} = \sum_a u_a \alpha_a u_a^T$


## Other properties

LMDPs have the property that if we have already solved two LMDPs, with the same state space, action space, unconditioned transition dynamics, but different state rewards, $q_1, q_2$. Then we can solve a new LMDP, again with the same, ..., and state rewards in the span of $q_1, q_2$, $z_3 = w_1 z_1 + w_2 z_2$, ...

Problem. What does it even mean for two LMDPs to have the same unconditioned dynamics but different state rewards?
The MDPs must have been the same up to some additive constant (constant in the actions), $r(s, a)=r(s, a) + c(s)$.
Does this really capture what we mean by different tasks?!?

AND HRL!?!?


Refs

- [Efficient computation of optimal actions](https://www.pnas.org/content/106/28/11478)
- [Linearly-solvable Markov decision problems](https://homes.cs.washington.edu/~todorov/papers/TodorovNIPS06.pdf)
- [Moving Least-squares Approximations for Linearly-solvable MDP](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5967383)
- [Aggregation Methods for Lineary-solvable Markov Decision Process](https://homes.cs.washington.edu/~todorov/papers/ZhongIFAC11.pdf)
- [A Unifying Framework for Linearly Solvable Control](https://arxiv.org/abs/1202.3715)
- [A Stability Result for Linear Markov Decision Processes](http://www.optimization-online.org/DB_FILE/2017/03/5893.pdf)
