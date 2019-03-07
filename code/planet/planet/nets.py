import jax.numpy as np
from jax import grad, jit, vmap
from jax.experimental.stax import serial, Dense, Relu, Softplus
from jax.experimental import optimizers

import collections

"""
TODO
- make a transition network with action abstraction
- a transition network with disentangled actions
- a multistep transition fn
"""

network = collections.namedtuple(
    'network',
    ['params', 'fn', 'outshape', 'loss_fn', 'grad_fn', 'step', 'opt_state'])

def mse(x, y):
    return np.mean(np.sum((x-y)**2, axis=-1))

def opt_update(i, net, batch):
    """
    Args:
        i (int): the step num
        net (network): a named tuple containing the parts of a neural network
        batch (tuple): the arguments to the loss fn - something like (inputs, targets)

    Returns:
        (network): a new network with updated parameters and optimiser state
    """
    new_opt_state = net.step(i, net.opt_state, batch)
    new_params = optimizers.get_params(new_opt_state)
    # TODO want sparse updating!? how slow is this?
    # maybe I shouldnt be using a named tuple for this?
    return network(new_params, net.fn, net.outshape, net.loss_fn,
                   net.grad_fn, net.step, new_opt_state)

def make_transition_net(n_inputs, n_actions, width, n_outputs):
    """
    Args:
        in_shape (tuple): (n_batch, n_inputs)
        width (int): the width of the network
        n_output (int): the number of dims in the output
    """
    init, fn = serial(
        Dense(width), Softplus,
        Dense(width), Softplus,
        Dense(width), Softplus,
        Dense(n_outputs)
    )

    out_shape, params = init((-1, n_inputs+n_actions))

    def apply_fn(params, x, a):
        x = np.concatenate([x,a],axis=-1)
        return fn(params, x)

    def loss_fn(params, x, a, t):
        # mean squared error
        return mse(apply_fn(params, x, a), t)

    # TODO jit
    dlossdparam = grad(loss_fn)

    opt_init, opt_update = optimizers.adam(step_size=0.001)
    opt_state = opt_init(params)

    def step(i, opt_state, batch):
        params = optimizers.get_params(opt_state)
        g = dlossdparam(params, *batch)
        return opt_update(i, g, opt_state)

    return network(params, jit(apply_fn) , out_shape, jit(loss_fn), jit(dlossdparam), jit(step), opt_state)

def make_value_net(n_inputs, width):
    init, fn = serial(
        Dense(width), Softplus,
        Dense(width), Softplus,
        Dense(width), Softplus,
        Dense(1)
    )

    out_shape, params = init((-1, n_inputs))

    def loss_fn(params, x_t, r_t, v_tp1, gamma):
        # mean squared bellman error
        v_t_approx = fn(params, x_t)
        v_t_target = r_t+gamma*v_tp1
        return mse(v_t_approx, v_t_target)

    # TODO jit
    dlossdparam = grad(loss_fn)

    opt_init, opt_update = optimizers.momentum(step_size=0.001, mass=0.9)
    opt_state = opt_init(params)

    def step(i, opt_state, batch):
          params = optimizers.get_params(opt_state)
          g = dlossdparam(params, *batch)
          return opt_update(i, g, opt_state)

    return network(params, jit(fn) , out_shape, jit(loss_fn), jit(dlossdparam), jit(step), opt_state)
