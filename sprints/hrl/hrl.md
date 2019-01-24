
Why would we want more depth? (greater length of time deps?? which is related to state size!?)





- Model based HRL
- Why is it easier to learn in a more temporally abstract space?
- Tensor HRL



***

Learning complexity

Assuming we have a subgoal space $G$ that covers the possible action space $\mid A\mid^k$.
Want $\mid G \mid \le \mid A\mid^k$.


Correlate rewards with;
- length k options/subgoals. then we have $\mathcal O(\mid A \mid^k)$ possible sequences. (not all actions are possible in all states so willnot be tight)
- set goal space to be state space.

we have factored out the redundancies of the many possible ways to get from one state to another.


(the argument is that we have reduced the size of the search space, thus making it easier to ??? what does search space have to do with reward?)
