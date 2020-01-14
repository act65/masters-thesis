import jax.numpy as np
import numpy.random as rnd
from jax import grad, jit, vmap
import matplotlib.pyplot as plt
import copy

# def loss_fn(x, t=3):  # mse
#     return (t-x)**2

def loss_fn(x):  # sigmoid cross entropy
    return -np.log(1/(1+np.exp(-x)))



# x = np.linspace(-3, 3, 100)
# plt.plot(x, loss_fn(x))
# plt.show()

reparams = [
    lambda z: z,
    lambda z: 2*z,
    lambda z: 1/z,

    # lambda x: 1/x + x,
    lambda z: np.exp(z),
    # lambda x: np.exp(x) + x,
    lambda z: np.exp(-z),
    # lambda x: np.exp(-x) + x,
    lambda z: np.log(z),
    lambda z: z**3,
]

inverses = [
    lambda x: x,
    lambda x: x/2,
    lambda x: 1/x,
    # lambda x: 1/x + x,
    lambda x: np.log(x),
    # lambda x: np.exp(x) + x,
    lambda x: -np.log(x),
    # lambda x: np.exp(-x) + x,
    lambda x: np.exp(x),
    lambda x: x**(1/3),
]

names = [
    'id',
    '2z',
    # '1/x + x',
    '1/z',
    'exp z',
    # 'exp x + x',
    'exp -z',
    # 'exp -x + x'
    'log z',
    'z^3',
]

def compose(f, g):
    return lambda z: f(g(z))

grad_fns = [grad(compose(loss_fn, f)) for f in reparams]

lr = 1e-2

errs = []
init_x = 10*rnd.random()
for f, g, inv in zip(reparams, grad_fns, inverses):
    z = inv(copy.deepcopy(init_x))
    losses = []
    for i in range(100):
        # if i == 0 or i % 10 == 0:
        losses.append(loss_fn(f(z)))
        z -= lr*g(z)
    errs.append(losses)

errs = np.array(errs)
n, m = errs.shape  # n fns, m iterations
# for i in range(m):
# plt.subplot(2, m//2, i+1)
# plt.bar(range(n), errs[:, i])
for i in range(n):
    plt.plot(errs[i, :], label=names[i])
# plt.xticks(range(n), names, rotation=70)
plt.legend()
plt.ylabel('Loss')
plt.xlabel('Iterations')
plt.savefig('pictures/figures/reparam-ce-04.png')
plt.show()
