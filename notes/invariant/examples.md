### Cart pole

#### Mirror symmetry

![Each pair is similar, in a sense](../../pictures/drawings/cart-pole-mirror.png)
<!-- What is this mirror around? Where is it? -->

Two indicators of this symmetry. It is reflected in the transition function, and the value function.

The change in state is conserved between the pair.

\begin{align}
\Delta_{\tau}(s, a) = \mathop{\mathbb E}_{s' \sim p(\cdot| s, a)} (s' - s) \\
\Delta_{\tau}(s_1, a_1) = - \Delta_{\tau}(s_2, a_2) \\
\end{align}

\begin{align}
\Delta_{T}(s, a) = (T \circ Q)(s,a) - Q(s,a)\\
\Delta_{T}(s_1, a_1) = \Delta_{T}(s_2, a_2) \\
\end{align}


The expected value is conserved between the pair (assuming we have a policy with mirror symmetry).

\begin{align}
\forall a \text{ set}\;\;\pi(a | s_1) = \pi(-a| s_2) \\
\forall \gamma: Q^\gamma_\pi(s_1, a_1) = Q^\gamma_\pi(s_2, a_2) \\
\forall \gamma: Q^\gamma_\pi(s_1, a_2) = Q^\gamma_\pi(s_2, a_1) \\
\end{align}

The (discounted) reachable rewards are conserved between the pair. (!!!)

\begin{align}
\{r(s, a, s'): \forall s \in \mathcal R(s_1, a_1)\} = \{r(s, a, s'): \forall s \in \mathcal R(s_2, a_2)\}
\end{align}

#### Translational symmetry

![Each pair is similar, in a sense](../../pictures/drawings/cart-pole-translation.png)

(special case of regular actions)


\begin{align}
\Delta(s_1, a_1) = \Delta(s_1, a_2) = \Delta(s_2, a_1) = \Delta(s_2, a_2) \\
\forall a \text{ set}\;\;\pi(a | s_1) = \pi(a| s_2) \\
\forall \gamma: Q^\gamma_\pi(s_1, a_1) = Q^\gamma_\pi(s_2, a_2)= Q^\gamma_\pi(s_1, a_2) = Q^\gamma_\pi(s_2, a_1), \\
\end{align}

### Local (approximate) symmetry


$$
\begin{align}
\Delta(s_1, a_1) = \Delta(s_1, a_2) = \Delta(s_2, a_1) = \Delta(s_2, a_2) \\
\forall a \text{ set}\;\;\pi(a | s_1) = \pi(a| s_2) \\
\forall \gamma: Q^\gamma_\pi(s_1, a_1) \approx Q^\gamma_\pi(s_2, a_2)  \\
\end{align}
$$

### Future translational symmetry

![Each pair is similar, in a sense](../../pictures/drawings/cart-pole-state.png)

different states, different actions. but maps into translational symmetry.

After this action. All future actions will have the same effect. In this sense, these two state-actions are similar.

\begin{align}
\forall a: \mathop{\mathbb E}_{s' \sim p(\cdot| s_1, a_1)} [\Delta(s', a)] =  \mathop{\mathbb E}_{s' \sim p(\cdot| s_2, a_2)} [\Delta(s', a)] \\
\end{align}

#### Temporal mirror symmetry

This is simply a result of the earlier mirror symmetry?!? (want to show this!)

permutations of actions that yield similar outcomes.

![Each pair is similar, in a sense](../../pictures/drawings/cart-pole-temporal-mirror.png)


\begin{align}
p(s'|s, \omega) = \prod p(s|s, a)\omega(a|s) \\
p(\cdot|s_1, \omega_1) = p(\cdot|s_1, \omega_2) \\
Q(s_1, \omega_1) = Q(s_1, \omega_2) \\
\end{align}

### Pong

#### Mirror symmetry (vertical)

![](../../pictures/drawings/pong-vert-flip.png)


\begin{align}
\Delta_{\tau}(s, a) = \mathop{\mathbb E}_{s' \sim p(\cdot| s, a)} (s' - s) \\
\Delta_{\tau}(s_1, a_1) = - \Delta_{\tau}(s_2, a_2) \\
\end{align}




\begin{align}
\Delta_{T}(s, a) = (T \circ Q)(s,a) - Q(s,a)\\
\Delta_{T}(s_1, a_1) = \Delta_{T}(s_2, a_2) \\
\end{align}


The expected value is conserved between the pair (assuming we have a policy with mirror symmetry).


\begin{align}
\forall a \text{ set}\;\;\pi(a | s_1) = \pi(-a| s_2) \\
\forall \gamma: Q^\gamma_\pi(s_1, a_1) = Q^\gamma_\pi(s_2, a_2) \\
\forall \gamma: Q^\gamma_\pi(s_1, a_2) = Q^\gamma_\pi(s_2, a_1) \\
\end{align}


(do these invariances uniquely define this mirror symmery?!)

#### Mirror symmetry (horizontal)

![](../../pictures/drawings/pong-horz-flip.png)


\begin{align}
Q(s_1, a_1) = - Q(s_2, a_2)
\end{align}


#### Translational symmetry
(although there are boundary cases which cannot be ignored. how can they be dealt with!?)

![](../../pictures/drawings/pong-trans.png)


\begin{align}
\forall a: Q(s_1, a) = Q(s_2, a)
\end{align}



#### Temporal symmetries

![](../../pictures/drawings/pong-reach.png)

\begin{align}
\exists \pi_1, \pi_2 \;\;\text{s.t.} \;\; Q^{\pi_1}(s_1, a_1) = Q^{\pi_2}(s_2, a_2)
\end{align}


The same future state can be reached, and thus the same rewards can be achieved.
