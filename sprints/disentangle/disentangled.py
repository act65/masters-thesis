import os
import numpy as np
import urllib

from absl import flags

import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

flags.DEFINE_float(
    "learning_rate", default=0.001, help="Initial learning rate.")
flags.DEFINE_integer(
    "epochs", default=100, help="Number of training steps to run.")
flags.DEFINE_string(
    "activation",
    default="selu",
    help="Activation function for all hidden layers.")
flags.DEFINE_integer(
    "batch_size",
    default=32,
    help="Batch size.")
flags.DEFINE_integer(
    "n_hidden",
    default=32,
    help="number of nodes in latent layer")
flags.DEFINE_integer(
    "width",
    default=64,
    help="width of intermediate layers")
flags.DEFINE_string(
    "data_dir",
    default="/tmp/mnist",
    help="Directory where data is stored (if using real data).")
flags.DEFINE_string(
    "model_dir",
    default="/tmp/test/",
    help="Directory to put the model's fit.")
flags.DEFINE_integer(
    "viz_steps", default=500, help="Frequency at which to save visualizations.")
flags.DEFINE_bool(
    "delete_existing",
    default=False,
    help="If true, deletes existing `model_dir` directory.")

FLAGS = flags.FLAGS

class Predictability():
    def __init__(self, n_dimensions):
        # could just use dropout instead!?
        self.predictor = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation=tf.nn.relu),
            tf.keras.layers.Dense(32, activation=tf.nn.relu),
            tf.keras.layers.Dense(n_dimensions)
        ])

    def __call__(self, x):
        x_ = tf.nn.dropout(x, 0.5)
        y = self.predictor(x_)
        return tf.losses.mean_squared_error(y, x)

def model_fn(features, labels, mode, params, config):
    """
    Builds the model function for use in an estimator.
    Arguments:
        features: The input features for the estimator.
        labels: The labels, unused here.
        mode: Signifies whether it is train or test or predict.
        params: Some hyperparameters as a dictionary.
        config: The RunConfig, unused here.
    Returns:
        EstimatorSpec: A tf.estimator.EstimatorSpec instance.
    """
    x = tf.layers.flatten(features['x'])

    global_step = tf.train.get_or_create_global_step()
    with tf.contrib.summary.record_summaries_every_n_global_steps(10, global_step=global_step):
        with tf.name_scope('build_models'):
            encoder = tf.keras.Sequential([
                tf.keras.layers.Dense(params['width'], activation=tf.nn.relu),
                tf.keras.layers.Dense(params['width'], activation=tf.nn.relu),
                tf.keras.layers.Dense(params['width'], activation=tf.nn.relu),
                tf.keras.layers.Dense(params['n_hidden']),
            ])

            # decoder = tf.keras.Sequential([
            #     tf.keras.layers.Dense(params['width'], activation=tf.nn.relu),
            #     tf.keras.layers.Dense(params['width'], activation=tf.nn.relu),
            #     tf.keras.layers.Dense(params['width'], activation=tf.nn.relu),
            #     tf.keras.layers.Dense(784),
            # ])

            pred = Predictability(params['n_hidden'])

        with tf.name_scope('build_graph'):
            h = encoder(x)
            loss = pred(h)
            # x_ = decoder(h)

            # recon = tf.losses.mean_squared_error(x_, x)

        with tf.name_scope('train'):
            opt = tf.train.AdamOptimizer()
            encoder_gnvs = opt.compute_gradients(-1e-4*loss, var_list=encoder.variables)
            pred_gnvs = opt.compute_gradients(loss, var_list=pred.predictor.variables)
            # decoder_gnvs = opt.compute_gradients(recon, var_list=decoder.variables)


            gnvs = encoder_gnvs+pred_gnvs
            gnvs = [(tf.clip_by_norm(g, 1.0), v) for g, v in gnvs]
            train_step = opt.apply_gradients(gnvs, global_step=global_step)

            # tf.linspace()
            # image_tile_summary('gen', )

    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_step,
      eval_metric_ops={"eval_loss": tf.metrics.mean(loss)}
    )


def pack_images(images, rows, cols):
  """Helper utility to make a field of images."""
  shape = tf.shape(images)
  width = shape[-3]
  height = shape[-2]
  depth = shape[-1]
  images = tf.reshape(images, (-1, width, height, depth))
  batch = tf.shape(images)[0]
  rows = tf.minimum(rows, batch)
  cols = tf.minimum(batch // rows, cols)
  images = images[:rows * cols]
  images = tf.reshape(images, (rows, cols, width, height, depth))
  images = tf.transpose(images, [0, 2, 1, 3, 4])
  images = tf.reshape(images, [1, rows * width, cols * height, depth])
  return images


def image_tile_summary(name, tensor, rows=8, cols=8):
  tf.summary.image(name, pack_images(tensor, rows, cols), max_outputs=1)


def main(_):
    params = FLAGS.flag_values_dict()
    params["activation"] = getattr(tf.nn, params["activation"])

    if FLAGS.delete_existing and tf.gfile.Exists(FLAGS.model_dir):
        tf.logging.warn("Deleting old log directory at {}".format(FLAGS.model_dir))
        tf.gfile.DeleteRecursively(FLAGS.model_dir)
    tf.gfile.MakeDirs(FLAGS.model_dir)

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
          x={"x": train_data},
          y=train_labels,
          batch_size=FLAGS.batch_size,
          num_epochs=1,
          shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        batch_size=FLAGS.batch_size,
        num_epochs=1,
        shuffle=False)

    estimator = tf.estimator.Estimator(
      model_fn,
      params=params,
      config=tf.estimator.RunConfig(
          model_dir=FLAGS.model_dir,
          save_checkpoints_steps=FLAGS.viz_steps,
      ),
    )

    for _ in range(FLAGS.epochs):
        estimator.train(train_input_fn, steps=FLAGS.viz_steps)
        eval_results = estimator.evaluate(eval_input_fn)
        print("Evaluation_results:\n\t%s\n" % eval_results)

if __name__ == "__main__":
    tf.app.run()
