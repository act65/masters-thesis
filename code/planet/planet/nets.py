import jax.numpy as np
from jax import grad, jit, vmap
from functools import partial

import jax.random as random

from jax.experimental.stax import serial, Dense, Relu
key = random.PRNGKey(0)

def make_transition_net(in_shape, width, n_output):
    # Use stax to set up network initialization and evaluation functions
    init, fn = serial(
        Dense(width), Relu,
        Dense(width), Relu,
        Dense(width), Relu,
        Dense(n_output)
    )

    out_shape, params = init(in_shape)

    def apply_fn(params, x, a):
        x = np.concatenate([x,a],axis=-1)
        return fn(params, x)

    def loss_fn(params, x, a, t):
        return np.mean(np.sum(apply_fn(params, x, a)**2 - t, axis=-1))

    # TODO jit
    dlossdparam = grad(loss_fn)

    return params, apply_fn , out_shape, loss_fn, dlossdparam


if __name__ == "__main__":
    params, fn, out_shape, loss_fn, dldp = make_transition_net((-1,4+2), 32, 4)

    key, subkey = random.split(key)
    x = random.normal(key, shape=(1, 4))
    key, subkey = random.split(key)
    a = random.normal(key, shape=(1, 2))
    key, subkey = random.split(key)
    t = random.normal(key, shape=(1, 4))

    y = fn(params, x, a)
    g = dldp(params, x, a, t)
    loss = loss_fn(params, x, a, t)
    assert y.shape == (1, 4)
