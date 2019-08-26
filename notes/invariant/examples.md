### Q values of a symmetric maze.

If we discover the maze's symmetry. We can use this to reduce to search space to $\frac{|S||A|}{2}$.

### Regular actions

If we discover that the four actions within the maze, `left`, `right`, `up`, `down`, are conserved over every state, we can reduce the search from $|S||S||A| \to |S| \times |A|$. (as sometimes the actions will result in null, because a wall is present. so we need to test each action in all possible states).

(this isnt true in many other cases. e.g. cart pole taking the same action doesnt result in the same change. at least not in the basis we are given.)

###


Pictures!!!


- Straight line. Moving a step towards our goal. These two states are similar in the sense that ???
- 
