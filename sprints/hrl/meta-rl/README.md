> 2. Meta-RL [@Wang2017LearningTR] trains a learner on the aggregated return over many episodes (a larger time scale than typical). If we construct a temporal decomposition (moving averages at different time-scales) of rewards and approximate them with a set of value functions, does this naturally produce a rich set of options (/hierarchical RL)?

- motivate the idea as a solution to an existing problem,
!?
- prove that the "existing" problem really exists,
!?

> Relationship between learning to learn and HRL?


Different speeds of learning yield meta learning!? Fast-slow weights. Memory. Worker-manager. ...? Predicting weights.

May as well call reptile/MAML fancy averaging...

But if each layer is exploring to estimate its own value and optimal policy. Then how can higher layer account for the extra variance!?

#### Ensemble of critics. Value decomposition (in temporal scale)

Each receiving different inputs?
Or could use fourier TD to estimate. Then we can reover an FFT!?
But what else can it represent? What can vanilla TD not represent? (oscillations!?)

Relationship to something like Rudder!?
