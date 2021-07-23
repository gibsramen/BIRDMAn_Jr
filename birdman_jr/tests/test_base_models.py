import unittest
import numpy as np
from scipy.special import rel_entr
from skbio.stats.composition import closure
from birdman_jr.base_models import (poisson_lognormal,
                                    negative_binomial,
                                    dirichlet_multinomial)
from birdman_jr.base_models import (input_matrix_validation,
                                    output_matrix_validation)


class TestBaseModels(unittest.TestCase):

    def setUp(self):
        self.mat = np.array([[24, 28, 98, 0, 0, 0],
                             [11, 20, 59, 0, 0, 0],
                             [139, 15, 46, 3, 0, 0],
                             [0, 0, 1, 18, 13, 295],
                             [0, 0, 0, 66, 137, 37],
                             [0, 0, 0, 29, 125, 83]])
        self.mat_zero = self.mat.copy()
        self.mat_zero[0, :] = 0
        self.cats = [1, 1, 1, 0, 0, 0]
        self.depths = self.mat.sum(1).reshape(self.mat.shape[0], -1)
        self.depths_ve1 = self.depths.copy()
        self.depths_ve1[0] = 0
        self.depths_ve2 = list(self.depths.copy())
        self.depths_ve3 = self.depths.copy()[:3, :]
        pass

    def test_poisson_lognormal(self):
        pln_mat = poisson_lognormal(self.mat, self.depths)[0]
        kldiv = rel_entr(closure(self.mat),
                         closure(pln_mat))
        kldiv[~np.isfinite(kldiv)] = 0.0
        self.assertTrue(0 <= kldiv.sum(1).max() <= 2)

    def test_negative_binomial(self):
        nm_mat = negative_binomial(self.mat, self.depths)[0]
        kldiv = rel_entr(closure(self.mat),
                         closure(nm_mat))
        kldiv[~np.isfinite(kldiv)] = 0.0
        self.assertTrue(0 <= kldiv.sum(1).max() <= 2)

    def test_dirichlet_multinomial(self):
        dm_mat = dirichlet_multinomial(self.mat, self.depths)[0]
        kldiv = rel_entr(closure(self.mat),
                         closure(dm_mat))
        kldiv[~np.isfinite(kldiv)] = 0.0
        self.assertTrue(0 <= kldiv.sum(1).max() <= 2)

    def test_input_matrix_validation_d1(self):
        with self.assertRaises(ValueError):
            input_matrix_validation(self.mat,
                                    self.depths_ve1)

    def test_input_matrix_validation_d2(self):
        with self.assertRaises(TypeError):
            input_matrix_validation(self.mat,
                                    self.depths_ve2)

    def test_input_matrix_validation_d3(self):
        with self.assertRaises(ValueError):
            input_matrix_validation(self.mat,
                                    self.depths_ve3)

    def test_output_matrix_validation(self):
        mat_res = output_matrix_validation(self.mat_zero)
        self.assertTrue(np.min(mat_res[0].shape)
                        < np.min(self.mat_zero.shape))
