Extensions.

- regression reachability
- policy conditioned on memory (via graph NN?)
- could do with a random similarity measure?!

Thoughts

- How is the memory related to the transition function!?
- Want to build a map. But what properties does a map have? Represents a fn. Has local and global structure! (can reason about local) -- and how does this relate to memory?
- Because we reset the episodic memory. there will always be exploration rewards at the start. good or bad!? OHH! this is actually quite important. it means we still have a chance to revisit some places, and get reward, again. (despite its lack of novelty, we just forgot it...) this avoids the problem of detachment ??? - see https://eng.uber.com/go-explore/
