### Estimating Q values of a symmetric maze.

If we discover the maze's symmetry. We can use this to reduce to search space to $\frac{|S||A|}{2}$.


### Estimating the transition fn.

If we discover that the four actions within the maze, `left`, `right`, `up`, `down`, are conserved over every state, we can reduce the search from $|S||S||A| \to |S| \times |A|$. (as sometimes the actions will result in null, because a wall is present. so we need to test each action in all possible states).
