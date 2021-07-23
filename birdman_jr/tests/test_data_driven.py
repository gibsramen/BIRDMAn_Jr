import unittest
import numpy as np
from biom import Table
from scipy.special import rel_entr
from skbio.stats.composition import closure
from birdman_jr.base_models import (poisson_lognormal,
                                    negative_binomial,
                                    dirichlet_multinomial)
from birdman_jr.data_driven import simulate


class TestDataDriven(unittest.TestCase):

    def setUp(self):
        self.mat = np.array([[24, 28, 98, 0, 0, 0],
                             [11, 20, 59, 0, 0, 0],
                             [139, 15, 46, 3, 0, 0],
                             [0, 0, 1, 18, 13, 295],
                             [0, 0, 0, 66, 137, 37],
                             [0, 0, 0, 29, 125, 83]])
        self.depths = self.mat.sum(1).reshape(self.mat.shape[0], -1)
        self.sids = ['s%i' % i for i in range(self.mat.shape[1])]
        self.fids = ['o%i' % i for i in range(self.mat.shape[0])]
        self.bt_test = Table(self.mat.T, self.fids, self.sids)

    def test_models_pln(self):
        bt_res = simulate(self.bt_test,
                          self.depths)
        mat_res = bt_res.matrix_data.toarray()
        mat_test = poisson_lognormal(self.mat,
                                     self.depths)[0]
        kldiv = rel_entr(closure(mat_test),
                         closure(mat_res))
        kldiv[~np.isfinite(kldiv)] = 0.0
        self.assertTrue(0 <= kldiv.sum(1).max() <= 10)

    def test_models_nb(self):
        bt_res = simulate(self.bt_test,
                          self.depths,
                          distribution='nb')
        mat_res = bt_res.matrix_data.toarray()
        mat_test = negative_binomial(self.mat,
                                     self.depths)[0]
        kldiv = rel_entr(closure(mat_test),
                         closure(mat_res))
        kldiv[~np.isfinite(kldiv)] = 0.0
        self.assertTrue(0 <= kldiv.sum(1).max() <= 10)

    def test_models_dm(self):
        bt_res = simulate(self.bt_test,
                          self.depths,
                          distribution='dm')
        mat_res = bt_res.matrix_data.toarray()
        mat_test = dirichlet_multinomial(self.mat,
                                         self.depths)[0]
        kldiv = rel_entr(closure(mat_test),
                         closure(mat_res))
        kldiv[~np.isfinite(kldiv)] = 0.0
        self.assertTrue(0 <= kldiv.sum(1).max() <= 10)
