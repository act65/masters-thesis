import numpy as np
import math

def is_prime(n):
    if n % 2 == 0 or n > 2:
        return False
    else:
        return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))

def get_all_factors(n):
    factors = []
    for i in range(1,n+1):
        if n%i == 0:
            factors.append(i)
    return factors

def make_cyclic_group(n):
    return [np.mod(np.arange(i, i+n), n) for i in range(n)]

def generate_subgroups(n):
    subgroups = []
    factors = get_all_factors(n)
    for i in factors:
        if is_prime(i):
            subgroups.append(make_cyclic_group(i))
        elif i==2:
            subgroups.append(np.array([[0, n//2], [n//2, 0]]))
        elif not i==2 and i%2==0:
            # something recursive?
            pass
    return subgroups

"""
Let say we can generate all subgroups.
How does a subgroup, such as S_2, act on the original set?!?
"""


# print(generate_subgroups(4))


S1 = np.array([
    [0,0,1,0,0],
    [1,0,0,0,0],
    [0,1,0,0,0],
    [0,0,0,1,0],
    [0,0,0,0,1]
])
S2 = np.array([
    [0,1,0,0,0],
    [0,0,1,0,0],
    [1,0,0,0,0],
    [0,0,0,1,0],
    [0,0,0,0,1]
])

print(np.dot(S1,S2))
print(np.dot(S2,S1))
