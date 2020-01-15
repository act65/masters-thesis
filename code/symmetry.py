import numpy as np

def symmetry_measure(x):
    """

    """
    n = x.size

    ### calculate which transformation the objects are invariant to
    permutations = np.stack(np.eye(n)[list(itertools.permutations(range(n), n)), :])

    diffs = x - np.einsum('ijk,kl->ijl', permutation, x)
    errs = np.linalg.norm(diffs, 'inf', axis=(1,2))  # shape = [n_permutations]
    # take the inf norm, as the objects are invariant, if ALL objects are invariant

    ### calculate how close the invariant tansforms are to a subgroup
    weights = sigmoid(-errs)  # weights. close to 1 if err is close to zero
    composition = np.einsum('ijl,ilk->ijk', permutations, permutations)
    distances =  np.linalg.norm(permutations[:, None, :, :]-composition[None, :, :, :], axis=(2,3))
    loss = np.dot(weights, distances)

    return loss


"""

argmax |H|
such that H\subset G; \sum_{g\in H} g \circ Q = Q

relaxation

Find nearest.
argmax_{\alpha} | \alpha | - \sum \alpha \parallel g_i \circ Q - Q \parallel - {[\alpha ; \alpha]}_{closure}
Find allocation of permutations that Q is invariant to, and that form a closed group.

Measure distance.
\epsilon_i = \parallel g_i \circ Q - Q \parallel
dist = {[\epsilon \cdot G ; \epsilon \cdot G]}_{closure}
"""

def symmetry_measure(x):
    # what is the complexity of this!?
    return min([invariance(subgroup, x) for subgroup in all_subgroups(G)])

def symmetry_measure(x):
    # what is the complexity of this!?
    # this wont work. wont be isomorphic, unless when clip the similarities to be in {0 ,1}.
    return min([is_isomorphic(subgroup, construct_graph(x)) for subgroup in all_subgroups(G)])

# NEED to reread the MDP homomorphism papers!

def main():

    # |S| x |A| -- BAD
    state_action_values = Q.reshape((-1, 1))
    symmetry_measure(state_action_values)

    # two state actions are related based on Q = r + \gamma P\piQ
    G = T(Q).ravel()[:, None] - Q.ravel()[None, :]  # |S||A| x |S||A|.
    # if we can relabel two state actions and G remains the same, then we have found a symmetry

    # wait a minute. when pi = \pi^{* } then we have T(Q) - Q = 0

    # what about that bellman rank!??!
