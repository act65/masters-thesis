import jax.numpy as np
from jax import grad, jit, vmap
from jax.experimental.stax import serial, Dense, Relu, Softplus, Tanh, Softmax
from jax.experimental import optimizers

import collections

"""
TODO
- make a transition network with action abstraction
- a transition network with disentangled actions
- a multistep transition fn
- prune the networks and augment the topolgy
- allow extra actions to be added
- ?
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

def make_transition_net(n_inputs, n_actions, width, n_outputs, activation=Relu):
    """
    Args:
        in_shape (tuple): (n_batch, n_inputs)
        width (int): the width of the network
        n_output (int): the number of dims in the output
    """
    init, fn = serial(
        Dense(width), activation,
        Dense(width), activation,
        Dense(width), activation,
        Dense(width), activation,
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

    opt_init, opt_update = optimizers.adam(step_size=1e-3)
    opt_state = opt_init(params)

    def step(i, opt_state, batch):
        params = optimizers.get_params(opt_state)
        g = dlossdparam(params, *batch)
        return opt_update(i, g, opt_state)

    return network(params, jit(apply_fn) , out_shape, jit(loss_fn), jit(dlossdparam), jit(step), opt_state)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

def make_value_net(n_inputs, width, activation=Relu):
    init, fn = serial(
        Dense(width), activation,
        Dense(width), activation,
        Dense(width), activation,
        Dense(width), activation,
        Dense(1)
    )

    out_shape, params = init((-1, n_inputs))

    # BUG  want to learn the value of the optimal policy! not fit to the current policy
    # TODO need off policy corrections!
    def loss_fn(params, x_t, r_t, v_tp1, gamma, a_t, a_logits):
        # mean squared bellman error - with importance sampled correction
        # delta = v(x_t) - rho gamma max_a [ r_t + v(t(s, a)) ]
        p = softmax(a_logits)
        rho = p[np.argmax(a_logits, axis=-1)] / p[np.argmax(a_t, axis=-1)]  # p = pi / b
        v_t_approx = fn(params, x_t)
        v_t_target = rho*(r_t+gamma*v_tp1)
        return mse(v_t_approx, v_t_target)

    # TODO jit
    dlossdparam = grad(loss_fn)

    opt_init, opt_update = optimizers.adam(step_size=1e-3)
    opt_state = opt_init(params)

    def step(i, opt_state, batch):
          params = optimizers.get_params(opt_state)
          g = dlossdparam(params, *batch)
          return opt_update(i, g, opt_state)

    return network(params, jit(fn) , out_shape, jit(loss_fn), jit(dlossdparam), jit(step), opt_state)

def whiten(x):
    return (x - np.mean(x, axis=-1, keepdims=True))/np.sqrt(np.var(x, axis=-1, keepdims=True) + 1e-8)

def a2c(logits, advantage):
    return -np.mean(logits * whiten(advantage))

def entropy(logits):
    p = softmax(logits)
    return - np.mean(np.sum(p * logits, axis=-1))

def make_actor_critic(n_inputs, width, n_actions, activation=Relu):
    init, fn = serial(
        Dense(width), activation,
        Dense(width), activation,
        Dense(width), activation,
        Dense(width), activation,
        Dense(n_actions+1)
    )

    out_shape, params = init((-1, n_inputs))

    def apply_fn(params, x):
        # NOTE so what is the relationship between the actions and the values?
        # this effectively learns Q logits?? QUESTION how is actor critic the same as Q?
        y = fn(params, x)
        return y[:, :-1], y[:,-1:]

    def loss_fn(params, x_t, r_t, v_tp1, gamma, a_t, a_logits):
        # off policy value correction
        p = softmax(a_logits)
        rho = p[np.argmax(a_logits, axis=-1)] / p[np.argmax(a_t, axis=-1)]  # p = pi / b
        v_t_target = rho*(r_t+gamma*v_tp1)

        v_t_approx, a_logits_approx = apply_fn(params, x_t)

        # mean squared bellman error
        value_loss = mse(v_t_approx, v_t_target)

        # soft advantage actor critic
        policy_loss = a2c(a_logits_approx[np.argmax(a_t, axis=-1)], v_t_target)
        policy_loss += -1e-2*entropy(a_logits_approx)

        return value_loss + policy_loss

    # TODO jit
    dlossdparam = grad(loss_fn)

    opt_init, opt_update = optimizers.adam(step_size=1e-4)
    opt_state = opt_init(params)

    def step(i, opt_state, batch):
          params = optimizers.get_params(opt_state)
          g = dlossdparam(params, *batch)
          return opt_update(i, g, opt_state)

    return network(params, jit(apply_fn) , out_shape, jit(loss_fn), jit(dlossdparam), jit(step), opt_state)
