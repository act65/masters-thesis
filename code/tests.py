import numpy as np
import utils



def test_rejection_sampler():
    class GaussianDist():
        def __init__(self, n, mu=None, stddev=None):
            self.mu = np.random.standard_normal((n, 1)) if mu is None else mu
            self.stddev = np.random.standard_normal((n, n)) if stddev is None else stddev

        def __call__(self, x):
            return gaussian_density(self.mu, self.stddev, x)

        def sample(self, x):
            return self.mu + (self.stddev**2) * np.random.standard_normal(self.stddev.shape)


    n = 2
    rejection_sample_include_prior(GaussianDist(n), GaussianDist(n, mu=0, stddev=1), k)
