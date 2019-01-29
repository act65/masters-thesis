

#### Multiscale state representation

If we had a multiscale state representation then we could build the policy as a fn of this representation.
Thus adding noise to the higher freq states would result in more local exploration (closer to random?!) and adding noise to the lower freq states would result in 'gobal' exploration over longer time periods!?

(huh, feels weird this has nothing to do with a heirarchical representation of the rewards)

#### Making interventions at various timescales.

There exist N different scales that we can apply interventions at. We want to know;
- what these interventions do
- which interventions lead to the highest reward

What does it mean to be at a different time scale? We get access to subsampled info, or it is averaged or ...!?
Or low/high pass filters? Or !?.
