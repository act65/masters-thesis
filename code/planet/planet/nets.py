import jax.numpy as np
from jax import grad, jit, vmap
from jax.experimental.stax import serial, Dense, Relu

import collections

"""
TODO
- make a transition network with action abstraction
- a transition network with disentangled actions
- a multistep transition fn
"""

network = collections.namedtuple(
    'network',
    ['params', 'fn', 'outshape', 'loss_fn', 'grad_fn'])

def mse(x, y):
    return np.mean(np.sum((x-y)**2, axis=-1))

def make_transition_net(in_shape, width, n_output):
    """
    Args:
        in_shape (tuple): (n_batch, n_inputs)
        width (int): the width of the network
        n_output (int): the number of dims in the output
    """
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
        # mean squared error
        return mse(apply_fn(params, x, a), t)

    # TODO jit
    dlossdparam = grad(loss_fn)

    return network(params, apply_fn , out_shape, loss_fn, dlossdparam)

def make_value_net(in_shape, width):
    init, fn = serial(
        Dense(width), Relu,
        Dense(width), Relu,
        Dense(width), Relu,
        Dense(1)
    )

    out_shape, params = init(in_shape)

    def loss_fn(params, x_t, r_t, v_tp1, gamma):
        # mean squared bellman error
        v_t_approx = fn(params, x_t)
        v_t_target = r_t+gamma*v_tp1
        return mse(v_t_approx, v_t_target)

    # TODO jit
    dlossdparam = grad(loss_fn)

    return network(params, fn , out_shape, loss_fn, dlossdparam)
