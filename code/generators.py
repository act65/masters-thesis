import numpy as np
import math
import matplotlib.pyplot as plt

n_states, n_actions = 4, 2

Q = np.array([
    [1,2],
    [3,4],
    [1,2],
    [3,4]
])

def pairwise_diff(x):
    return x.ravel()[None, :] - x.ravel()[:, None]

print(np.arange(8).reshape((4, 2)))
print(np.arange(8).reshape((4, 2)).ravel())

D = pairwise_diff(Q)
M = pairwise_diff(D)

plt.figure(figsize=(16,16))
plt.subplot(2, 1, 1)
plt.imshow(D==0)
plt.subplot(2,1,2)
plt.imshow(M==0)
plt.show()
# print((pairwise_diff(pairwise_diff(Q))==0))


"""
If; Q(s, a) - Q(s'', a'') = Q(s', a') - Q(s''', s''')
then we can conclude? There exists f such that
Q(s, a) - f(Q)(s, a) = Q(s', a') - f(Q)(s, a)
Which implies?
exists g_i := Q(s, a) - f(Q)(s, a)

Want to find relations between these g_i's?
first find unique g_i's: g_i = g_j  (aka the generators!?!?)
then find relations: g_i^2 = g_k^4 = (g_ig_k)^2 = e
finally, use generators and relations to construct group

now what... we have the group.
which action does it use?
if we have the action and the group,
then, we can average over orbits!?

costs (|S| \times |A|)^4 ???


But. What if we are dealing with estimates of the Qs?
We will have some uncertainty about the generators and the relations!?

Also. Nonlinearity could alias some of the differences? Might need higher orders?

"""
