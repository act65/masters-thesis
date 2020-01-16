import itertools
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

n = 5
P = np.stack(np.eye(n)[list(itertools.permutations(range(n), n)), :])

x = np.array([1.1,1.05,1,3,4])

errs = np.linalg.norm(x - np.einsum('ijk,k->ij', P, x), axis=1)
# sims = sigmoid(np.einsum('ij,j->i', np.einsum('ijk,k->ij', P, x), x)/np.linalg.norm(x))
sims = 1/(1+errs)

# print(P[])
g_idx = np.where(errs<0.5)
group = P[g_idx]

m = len(group)
G = np.stack(group)

# for i in range(m):
#     for j in range(m):
#         bool = np.dot(G[i], G[j]) in G
#         print('{}, {}: {}'.format(i, j, bool))

def dist_to_closest(x, T):
    diff = np.linalg.norm(x - T, axis=(1,2))
    idx = np.argmin(diff)
    return idx, np.linalg.norm(T[idx] - x)

# m^2. for the permutations that are close to the data. they should form a group.
for i in g_idx[0]:
    for j in g_idx[0]:
        idx, diff = dist_to_closest(np.dot(P[i], P[j]), P)
        print('{}, {}: {}'.format(i, j, sims[idx] * sims[i] * sims[j]))



print(sims[3] * sims[1] * sims[2])

# problem. because we threshold. there will be not gradients for all permutations we have cut out.
# want a kind of soft attention

# but want it to be fully differentiable!?
# need to construct a ijk tensor of the i o j => k transitions
# then we can collect all the errors?
