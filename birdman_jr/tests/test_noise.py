import unittest
import numpy as np
from skbio.stats.composition import closure
from numpy.testing import assert_array_almost_equal
from birdman_jr.noise import add_noise

class TestNoise(unittest.TestCase):

    def setUp(self):
        self.mat = np.array([[24, 28, 98, 0, 0, 0],
                             [11, 20, 59, 0, 0, 0],
                             [139, 15, 46, 3, 0, 0],
                             [0, 0, 1, 18, 13, 295],
                             [0, 0, 0, 66, 137, 37],
                             [0, 0, 0, 29, 125, 83]])

    def test_no_noise(self):
        add_nothing = add_noise(self.mat,
                                percent_normal=0,
                                percent_random=0,
                                add_missing_at_random=True,
                                percent_missing=0)
        self.assertTrue(np.allclose(add_nothing,
                                    closure(self.mat),
                                    atol=0.1))

    def test_add_missing(self):
        sparse_mat = add_noise(self.mat,
                                percent_normal=0,
                                percent_random=0,
                                add_missing_at_random=True,
                                percent_missing=0.35)
        self.assertTrue(round(np.count_nonzero(sparse_mat == 0) / 36, 1)
                        == 0.3)
