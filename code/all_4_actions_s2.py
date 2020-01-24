import numpy as np
import itertools

n = 3
P = np.stack(np.eye(n)[list(itertools.permutations(range(n), n)), :])

idx = np.where(np.all(np.eye(n) == np.einsum('ijl,ilk->ijk', P, P), axis=(1,2)))

print(len(idx[0]))
print(P[idx])
