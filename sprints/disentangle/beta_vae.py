# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Trains a variational auto-encoder (VAE) on binarized MNIST.

The VAE defines a generative model in which a latent code `Z` is sampled from a
prior `p(Z)`, then used to generate an observation `X` by way of a decoder
`p(X|Z)`. The full reconstruction follows

```none
   X ~ p(X)              # A random image from some dataset.
   Z ~ q(Z | X)          # A random encoding of the original image ("encoder").
Xhat ~ p(Xhat | Z)       # A random reconstruction of the original image
                         #   ("decoder").
```

To fit the VAE, we assume an approximate representation of the posterior in the
form of an encoder `q(Z|X)`. We minimize the KL divergence between `q(Z|X)` and
the true posterior `p(Z|X)`: this is equivalent to maximizing the evidence lower
bound (ELBO),

```none
-log p(x)
= -log int dz p(x|z) p(z)
= -log int dz q(z|x) p(x|z) p(z) / q(z|x)
<= int dz q(z|x) (-log[ p(x|z) p(z) / q(z|x) ])   # Jensen's Inequality
=: KL[q(Z|x) || p(x|Z)p(Z)]
= -E_{Z~q(Z|x)}[log p(x|Z)] - KL[q(Z|x) || p(Z)]
```

-or-

```none
-log p(x)
= KL[q(Z|x) || p(x|Z)p(Z)] - KL[q(Z|x) || p(Z|x)]
<= KL[q(Z|x) || p(x|Z)p(Z)                        # Positivity of KL
= -E_{Z~q(Z|x)}[log p(x|Z)] - KL[q(Z|x) || p(Z)]
```

The `-E_{Z~q(Z|x)}[log p(x|Z)]` term is an expected reconstruction loss and
`KL[q(Z|x) || p(Z)]` is a kind of distributional regularizer. See
[Kingma and Welling (2014)][1] for more details.

This script supports both a (learned) mixture of Gaussians prior as well as a
fixed standard normal prior. You can enable the fixed standard normal prior by
setting `mixture_components` to 1. Note that fixing the parameters of the prior
(as opposed to fitting them with the rest of the model) incurs no loss in
generality when using only a single Gaussian. The reasoning for this is
two-fold:

  * On the generative side, the parameters from the prior can simply be absorbed
    into the first linear layer of the generative net. If `z ~ N(mu, Sigma)` and
    the first layer of the generative net is given by `x = Wz + b`, this can be
    rewritten,

      s ~ N(0, I)
      x = Wz + b
        = W (As + mu) + b
        = (WA) s + (W mu + b)

    where Sigma has been decomposed into A A^T = Sigma. In other words, the log
    likelihood of the model (E_{Z~q(Z|x)}[log p(x|Z)]) is independent of whether
    or not we learn mu and Sigma.

  * On the inference side, we can adjust any posterior approximation
    q(z | x) ~ N(mu[q], Sigma[q]), with

    new_mu[p] := 0
    new_Sigma[p] := eye(d)
    new_mu[q] := inv(chol(Sigma[p])) @ (mu[p] - mu[q])
    new_Sigma[q] := inv(Sigma[q]) @ Sigma[p]

    A bit of algebra on the KL divergence term `KL[q(Z|x) || p(Z)]` reveals that
    it is also invariant to the prior parameters as long as Sigma[p] and
    Sigma[q] are invertible.

This script also supports using the analytic KL (KL[q(Z|x) || p(Z)]) with the
`analytic_kl` flag. Using the analytic KL is only supported when
`mixture_components` is set to 1 since otherwise no analytic form is known.

Here we also compute tighter bounds, the IWAE [Burda et. al. (2015)][2].

These as well as image summaries can be seen in Tensorboard. For help using
Tensorboard see
https://www.tensorflow.org/guide/summaries_and_tensorboard
which can be run with
  `python -m tensorboard.main --logdir=MODEL_DIR`

#### References

[1]: Diederik Kingma and Max Welling. Auto-Encoding Variational Bayes. In
     _International Conference on Learning Representations_, 2014.
     https://arxiv.org/abs/1312.6114
[2]: Yuri Burda, Roger Grosse, Ruslan Salakhutdinov. Importance Weighted
     Autoencoders. In _International Conference on Learning Representations_,
     2015.
     https://arxiv.org/abs/1509.00519
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Adapted from https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py
# TODO: use Daniel's rate limited VAE?
# TODO: verify how color is encoded in the sprites dataset. (I think there might be a bug in my code.)
# TODO: explore how the number of dims effects what is disentangled
# TODO: plot latent space, what do disentangled variables look like. does clustering make sense?
# python beta_vae.py --mixture_components=1 --analytic_kl=True --n_samples=1 --model_dir=/tmp/vae/0

import functools
import os

# Dependency imports
from absl import flags
import numpy as np
import h5py
from six.moves import urllib
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

IMAGE_SHAPE = [64, 64, 1]

flags.DEFINE_float(
    "learning_rate", default=0.001, help="Initial learning rate.")
flags.DEFINE_integer(
    "max_steps", default=50001, help="Number of training steps to run.")
flags.DEFINE_integer(
    "latent_size",
    default=16,
    help="Number of dimensions in the latent code (z).")
flags.DEFINE_integer("base_depth", default=32, help="Base depth for layers.")
flags.DEFINE_string(
    "activation",
    default="leaky_relu",
    help="Activation function for all hidden layers.")
flags.DEFINE_integer(
    "batch_size",
    default=32,
    help="Batch size.")
flags.DEFINE_integer(
    "n_samples", default=16, help="Number of samples to use in encoding.")
flags.DEFINE_integer(
    "mixture_components",
    default=100,
    help="Number of mixture components to use in the prior. Each component is "
         "a diagonal normal distribution. The parameters of the components are "
         "intialized randomly, and then learned along with the rest of the "
         "parameters. If `analytic_kl` is True, `mixture_components` must be "
         "set to `1`.")
flags.DEFINE_bool(
    "analytic_kl",
    default=False,
    help="Whether or not to use the analytic version of the KL. When set to "
         "False the E_{Z~q(Z|X)}[log p(Z)p(X|Z) - log q(Z|X)] form of the ELBO "
         "will be used. Otherwise the -KL(q(Z|X) || p(Z)) + "
         "E_{Z~q(Z|X)}[log p(X|Z)] form will be used. If analytic_kl is True, "
         "then you must also specify `mixture_components=1`.")
flags.DEFINE_string(
    "data_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/data"),
    help="Directory where data is stored (if using real data).")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "beta-vae/0"),
    help="Directory to put the model's fit.")
flags.DEFINE_integer(
    "viz_steps", default=100, help="Frequency at which to save visualizations.")
flags.DEFINE_bool(
    "delete_existing",
    default=False,
    help="If true, deletes existing `model_dir` directory.")
flags.DEFINE_float(
    "beta_init",
    default=40.0,
    help="The initial strength of regulartisation/rate limiting."
)

FLAGS = flags.FLAGS


def _softplus_inverse(x):
  """Helper which computes the function inverse of `tf.nn.softplus`."""
  return tf.log(tf.math.expm1(x))


def make_encoder(activation, latent_size, base_depth):
    """Creates the encoder function.

    Args:
    activation: Activation function in hidden layers.
    latent_size: The dimensionality of the encoding.
    base_depth: The lowest depth for a layer.

    Returns:
    encoder: A `callable` mapping a `Tensor` of images to a
      `tfd.Distribution` instance over encodings.
    """
    conv = functools.partial(
      tf.keras.layers.Conv2D, padding="SAME", activation=activation)

    encoder_net = tf.keras.Sequential([
      conv(base_depth, 5, 2),
      conv(base_depth, 5, 2),
      conv(2 * base_depth, 5, 2),
      conv(2 * base_depth, 5, 2),
      conv(4 * latent_size, 7, 2),
      tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(latent_size, activation=None),
    ])

    def encoder(images):
        images = 2 * tf.cast(images, dtype=tf.float32) - 1
        net = encoder_net(images)
        return tfd.MultivariateNormalDiag(
          loc=net,
          scale_diag=tf.ones_like(net),
          name="code")

    return encoder


def make_decoder(activation, latent_size, output_shape, base_depth):
  """Creates the decoder function.

  Args:
    activation: Activation function in hidden layers.
    latent_size: Dimensionality of the encoding.
    output_shape: The output image shape.
    base_depth: Smallest depth for a layer.

  Returns:
    decoder: A `callable` mapping a `Tensor` of encodings to a
      `tfd.Distribution` instance over images.
  """
  deconv = functools.partial(
      tf.keras.layers.Conv2DTranspose, padding="SAME", activation=activation)
  conv = functools.partial(
      tf.keras.layers.Conv2D, padding="SAME", activation=activation)

  decoder_net = tf.keras.Sequential([
      deconv(2 * base_depth, 7, 2),
      deconv(2 * base_depth, 5, 2),
      deconv(2 * base_depth, 5, 2),
      deconv(base_depth, 5, 2),
      deconv(base_depth, 5, 2),
      deconv(base_depth, 5, 2),
      conv(output_shape[-1], 5, activation=None),
  ])

  def decoder(codes):
    original_shape = tf.shape(codes)
    # Collapse the sample and batch dimension and convert to rank-4 tensor for
    # use with a convolutional decoder network.
    codes = tf.reshape(codes, (-1, 1, 1, latent_size))
    logits = decoder_net(codes)
    logits = tf.reshape(
        logits, shape=tf.concat([original_shape[:-1], output_shape], axis=0))
    return tfd.Independent(tfd.Bernoulli(logits=logits),
                           reinterpreted_batch_ndims=len(output_shape),
                           name="image")

  return decoder


def make_mixture_prior(latent_size, mixture_components):
  """Creates the mixture of Gaussians prior distribution.

  Args:
    latent_size: The dimensionality of the latent representation.
    mixture_components: Number of elements of the mixture.

  Returns:
    random_prior: A `tfd.Distribution` instance representing the distribution
      over encodings in the absence of any evidence.
  """
  if mixture_components == 1:
    # See the module docstring for why we don't learn the parameters here.
    return tfd.MultivariateNormalDiag(
        loc=tf.zeros([latent_size]),
        scale_identity_multiplier=1.0)

  loc = tf.get_variable(name="loc", shape=[mixture_components, latent_size])
  raw_scale_diag = tf.get_variable(
      name="raw_scale_diag", shape=[mixture_components, latent_size])
  mixture_logits = tf.get_variable(
      name="mixture_logits", shape=[mixture_components])

  return tfd.MixtureSameFamily(
      components_distribution=tfd.MultivariateNormalDiag(
          loc=loc,
          scale_diag=tf.nn.softplus(raw_scale_diag)),
      mixture_distribution=tfd.Categorical(logits=mixture_logits),
      name="prior")


def build_interpolated_latents(N, latent_size):
    """Interpolate, independently, across the different latent dimensions"""
    vals = tf.concat([tf.linspace(-6.0, 6.0, N)  for _ in range(latent_size)], axis=0)
    indices = tf.constant([i for j in range(latent_size) for i in N*[j]])
    ones = tf.one_hot(indices, latent_size, on_value=1.0, off_value=0.0)
    return tf.expand_dims(vals, -1)*ones


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


def model_fn(features, labels, mode, params, config):
    """Builds the model function for use in an estimator.

    Arguments:
    features: The input features for the estimator.
    labels: The labels, unused here.
    mode: Signifies whether it is train or test or predict.
    params: Some hyperparameters as a dictionary.
    config: The RunConfig, unused here.

    Returns:
    EstimatorSpec: A tf.estimator.EstimatorSpec instance.
    """
    del labels, config

    global_step = tf.train.get_or_create_global_step()
    beta = tf.train.polynomial_decay(
        params["beta_init"],
        global_step,
        params["max_steps"]//10,
        end_learning_rate=1.0,
        power=0.9)
    tf.summary.scalar('beta', beta)

    if params["analytic_kl"] and params["mixture_components"] != 1:
        raise NotImplementedError(
            "Using `analytic_kl` is only supported when `mixture_components = 1` "
            "since there's no closed form otherwise.")

    encoder = make_encoder(params["activation"],
                         params["latent_size"],
                         params["base_depth"])
    decoder = make_decoder(params["activation"],
                         params["latent_size"],
                         IMAGE_SHAPE,
                         params["base_depth"])
    latent_prior = make_mixture_prior(params["latent_size"],
                                    params["mixture_components"])

    image_tile_summary("input", tf.to_float(features), rows=1, cols=16)

    approx_posterior = encoder(features)
    approx_posterior_sample = approx_posterior.sample(params["n_samples"])
    decoder_likelihood = decoder(approx_posterior_sample)
    image_tile_summary(
      "recon/sample",
      tf.to_float(decoder_likelihood.sample()[:3, :16]),
      rows=3,
      cols=16)
    image_tile_summary(
      "recon/mean",
      decoder_likelihood.mean()[:3, :16],
      rows=3,
      cols=16)

    # `distortion` is just the negative log likelihood.
    distortion = -decoder_likelihood.log_prob(features)
    avg_distortion = tf.reduce_mean(distortion)
    tf.summary.scalar("distortion", avg_distortion)

    if params["analytic_kl"]:
        rate = tfd.kl_divergence(approx_posterior, latent_prior)
    else:
        rate = (approx_posterior.log_prob(approx_posterior_sample)
            - latent_prior.log_prob(approx_posterior_sample))
    avg_rate = tf.reduce_mean(rate)
    tf.summary.scalar("rate", avg_rate)

    elbo_local = -(beta*rate + distortion)

    elbo = tf.reduce_mean(elbo_local)
    loss = -elbo
    tf.summary.scalar("elbo", elbo)

    importance_weighted_elbo = tf.reduce_mean(
      tf.reduce_logsumexp(elbo_local, axis=0) -
      tf.log(tf.to_float(params["n_samples"])))
    tf.summary.scalar("elbo/importance_weighted", importance_weighted_elbo)

    # Decode samples from the prior for visualization.
    random_image = decoder(latent_prior.sample(16))
    image_tile_summary(
      "random/sample", tf.to_float(random_image.sample()), rows=4, cols=4)
    image_tile_summary("random/mean", random_image.mean(), rows=4, cols=4)

    # Generate interpolated values from the prior
    N = 16
    interp_z = build_interpolated_latents(N, params['latent_size'])
    interp_image = decoder(tf.expand_dims(interp_z, 0))
    image_tile_summary(
        "interp/sample", tf.to_float(interp_image.sample()), rows=params['latent_size'], cols=N)
    image_tile_summary("interp/mean", interp_image.mean(), rows=params['latent_size'], cols=N)

    # Perform variational inference by minimizing the -ELBO.
    learning_rate = tf.train.cosine_decay(params["learning_rate"], global_step,
                                        params["max_steps"])
    tf.summary.scalar("learning_rate", learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops={
          "elbo": tf.metrics.mean(elbo),
          "elbo/importance_weighted": tf.metrics.mean(importance_weighted_elbo),
          "rate": tf.metrics.mean(avg_rate),
          "distortion": tf.metrics.mean(avg_distortion),
      },
    )


class Generator():
    def __init__(self, fname, batch_size):
        self.batch_size = batch_size
        with h5py.File(fname, 'r') as hf:
            # not super nice. need to load the entire file into memory...
            # can load more efficiently with h5py but then shuffling becomes expensive
            self.imgs = np.array(hf['imgs'])

        self.latents_sizes = np.array([ 1,  3,  6, 40, 32, 32])
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                        np.array([1,])))

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def sample_latent(self, size=1):
        samples = np.zeros((size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)
        return samples

    def __call__(self):
            for _ in range(737280//self.batch_size):
                latents_sampled = self.sample_latent(size=self.batch_size)
                indices_sampled = self.latent_to_index(latents_sampled)
                yield self.imgs[indices_sampled]

def build_input_fns(batch_size):

    def train_input_fn():
        gen = Generator("/local/scratch/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5", batch_size)
        return tf.data.Dataset.from_generator(gen, tf.int32, tf.TensorShape([batch_size, 64, 64])).map(lambda x:tf.expand_dims(x,-1))

    return train_input_fn, train_input_fn

def main(argv):
    del argv  # unused

    params = FLAGS.flag_values_dict()
    params["activation"] = getattr(tf.nn, params["activation"])
    if FLAGS.delete_existing and tf.gfile.Exists(FLAGS.model_dir):
        tf.logging.warn("Deleting old log directory at {}".format(FLAGS.model_dir))
        tf.gfile.DeleteRecursively(FLAGS.model_dir)
    tf.gfile.MakeDirs(FLAGS.model_dir)

    train_input_fn, eval_input_fn = build_input_fns(FLAGS.batch_size)

    estimator = tf.estimator.Estimator(
      model_fn,
      params=params,
      config=tf.estimator.RunConfig(
          model_dir=FLAGS.model_dir,
          save_checkpoints_steps=FLAGS.viz_steps,
      ),
    )

    for _ in range(FLAGS.max_steps // FLAGS.viz_steps):
        estimator.train(train_input_fn, steps=FLAGS.viz_steps)
        # eval_results = estimator.evaluate(eval_input_fn)
        # print("Evaluation_results:\n\t%s\n" % eval_results)


if __name__ == "__main__":
    tf.app.run()
