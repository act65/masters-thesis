import numpy as np

# C4. cayley table
C4 = np.array([
    [0, 1, 2, 3],
    [1, 2, 3, 0],
    [2, 3, 0, 1],
    [3, 0, 1, 2],
])

# # C2 x C2. cayley table
# C2_2 = np.array([
#     [0, 1, 2, 3],
#     [1, 2, 3, 0],
#     [2, 3, 0, 1],
#     [3, 0, 1, 2],
# ])

C2 = np.array([
 [0, 1],
 [1, 0]
])

def direct_product(G0, G1):
    """
    H = G1 \times G2
    h_i = (a, x)
    h_j = (b, y)
    h_i \circ h_j = (a, x) \circ (b, y) = (a \circ b, x \circ y)
    """
    n = G0.shape[0]
    m = G1.shape[0]
    elems = [(i, j) for i in range(n) for j in range(m)]
    cayley = [[(G0[x[0], y[0]], G1[x[1], y[1]]) for x in elems] for y in elems]
    return np.array([[elems.index(e) for e in row] for row in cayley])

C2_2 = direct_product(C2, C2)
print(C2_2)
print(C4)

"""
Desiderata of complexity measure
- \mathcal L(C_4) > L(C2_2)


Complexity(x) = min_y [complexity of y + complexity of encoding the error (y - x) ]

note. if there is no encoding error, then we should pick the simplest y with x=y
it should not pick some other y' because it is 'kinda' close to , but has x-y' = e. and y' is very simple. or should it?
"""
