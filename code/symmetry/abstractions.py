import tensorflow as tf
tf.enable_eager_execution()

def cosine_similarity(x, y):
    # cosine similarity = 1 - cosine dist
    return 1-tf.losses.cosine_distance(x, y, axis=-1, reduction=tf.losses.Reduction.NONE)

class Abstraction():
    def __init__(self, n_inputs, n_hidden, width=32, activation=tf.nn.selu, depth=2, logdir='/tmp/abs/0'):
        self.n_inputs = n_inputs
        layers = [tf.keras.layers.Dense(width, activation=activation)] * depth
        self.net = tf.keras.Sequential(
            [tf.keras.layers.Dense(width, activation=activation, input_shape=(n_inputs,))]
            + layers +
            [tf.keras.layers.Dense(n_inputs, activation=tf.nn.sigmoid)]
            )
        self.variables = self.net.variables

        self.writer = tf.contrib.summary.create_file_writer(logdir)
        self.writer.set_as_default()


    def __call__(self, inputs):
        return self.net(inputs)

    def loss_fn(self, x_i, x_j, t):
        t = tf.expand_dims(t, 1)
        h_i, h_j = self.net(x_i), self.net(x_j)

        sim_ij = cosine_similarity(h_i, h_j)
        # NOTE could do all pairs. but this might be sufficient!?
        sim_ii = cosine_similarity(h_i[1:], h_i[:-1])
        sim_jj = cosine_similarity(h_j[1:], h_j[:-1])

        tf.contrib.summary.scalar('loss/sim_ij', tf.reduce_mean(sim_ij))
        tf.contrib.summary.scalar('loss/sim_ii', tf.reduce_mean(sim_ii))
        tf.contrib.summary.scalar('loss/sim_jj', tf.reduce_mean(sim_jj))

        tf.contrib.summary.histogram('similarities', t)

        # try to minimise the cosine similarity between representations
        # unless they are 'similar', given by t.
        return tf.reduce_mean((1-t) * sim_ij)

    def grads(self, x_i, x_j, t):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(x_i, x_j, t)
        return loss, tape.gradient(loss, self.variables)

    def train(self, gen, lr=1e-3, batch_size=50):
        opt = tf.train.AdamOptimizer(lr)
        ds = tf.data.Dataset.from_generator(
            lambda : gen, (tf.float32, tf.float32, tf.float32),
            (tf.TensorShape([self.n_inputs]), tf.TensorShape([self.n_inputs]), tf.TensorShape([])))
        ds = ds.shuffle(1000)
        ds = ds.batch(batch_size)

        iterator = ds.make_one_shot_iterator()

        with tf.contrib.summary.record_summaries_every_n_global_steps(10):
            for i, batch in enumerate(iterator):
                L, g = self.grads(*batch)
                print('\rStep: {}, Loss: {:.4f}'.format(i, L), end='', flush=True)
                opt.apply_gradients(zip(g, self.variables), global_step=tf.train.get_or_create_global_step())

if __name__ == '__main__':
    traj = [
        (tf.random.normal((8,)),
        tf.random.normal((8,)),
        tf.random.uniform(shape=(1,), minval=0.0, maxval=1.0))
        for _ in range(1000)]

    model = Abstraction(8, 4)
    model.train(traj)
