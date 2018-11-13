"""
Learn a simple nd energy function learned via observations of its trajectories.
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class EnergyDynamics(object):
    def __init__(self):
        self.energy_fn = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.sigmoid, input_shape=[1]),
            tf.keras.layers.Dense(128, activation=tf.nn.sigmoid),
            tf.keras.layers.Dense(1),
        ])

    def forward(self, x, step_size=1e-1):
        e = self.energy_fn(x)
        g = tf.gradients(e, x)[0]
        x_t = x - step_size * g
        return x_t

    def get_loss(self, x, x_t):
        return tf.losses.mean_squared_error(self.forward(x), x_t)

def plot(step, x, y, t):
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax.plot(x, y)
    ax.plot(x, t)
    fig.savefig('ims/test{}.png'.format(step))   # save the figure to file

def main():
    target = Target(target_fn)
    x, t = target.generate_batch(50)
    tf.logging.warn('{}, {}'.format(x.get_shape(), t.get_shape()))

    model = EnergyDynamics()
    loss = model.get_loss(x, x-0.1*t)

    X = tf.expand_dims(tf.linspace(-3.0, 3.0, 1000), 1)
    Y = model.energy_fn(X)
    T = target_fn(X)

    # center around zero.
    # matching the gradient doesnt mean the absolute values will match
    reg = tf.reduce_sum(tf.square(Y))
    loss += 1e-6*reg

    global_step = tf.train.get_or_create_global_step()
    train_step = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

    tf.summary.scalar('loss', loss)

    writer = tf.summary.FileWriter('/tmp/iel/0')
    summaries = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            if i%500 == 0:
                plot(*sess.run([global_step, X, Y, T]))

            summ, _, L = sess.run([summaries, train_step, loss])
            print('\rStep: {}, loss: {}'.format(i, L), end='', flush=True)
            writer.add_summary(summ, i)

if __name__ == "__main__":
    main()
