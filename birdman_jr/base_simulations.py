import numpy as np
from skbio.stats.composition import closure
from numpy.random import (poisson, lognormal, gamma,
                          dirichlet, multinomial)


def poisson_lognormal(mat, depths, kappa=1):

    """
    Simulate from counts, probabilities, or
    proportions of input matrix with a 
    Poisson Log-Normal distribution.

    mat: array_like
        matrix of strictly positive counts
        or probabilities/proportions.
        columns = features (components)
        rows = samples (compositions)
    depth : array_like
        Read depth of the simulation
        for each sample (row).
    """

    # check matrix and ensure
    # data is proportions
    mat = input_matrix_validation(mat, depths)
    # simulate from proportions
    mu = depths * mat
    sim = np.vstack([poisson(lognormal(np.log(mu[i, :]), kappa))
                     for i in range(mat.shape[0])])

    return output_matrix_validation(sim)

def negative_binomial(mat, depths, kappa=1):

    """
    Simulate from counts, probabilities, or
    proportions of input matrix with a 
    Negative Binomial distribution.

    mat: array_like
        matrix of strictly positive counts
        or probabilities/proportions.
        columns = features (components)
        rows = samples (compositions)
    depth : array_like
        Read depth of the simulation
        for each sample (row).
    """

    # check matrix and ensure
    # data is proportions
    mat = input_matrix_validation(mat, depths)
    # simulate from proportions
    mu = depths * mat
    sim = np.vstack([poisson(gamma(kappa, kappa * mu[i, :]))
                     for i in range(mat.shape[0])])

    return output_matrix_validation(sim)

def dirichlet_multinomial(mat, depths,
                          dirichlet=False,
                          pseudocount=0.001):

    """
    Simulate from counts, probabilities, or
    proportions of input matrix with a 
    Dirichlet Multinomial distribution.

    mat: array_like
        matrix of strictly positive counts
        or probabilities/proportions.
        columns = features (components)
        rows = samples (compositions)
    depth : array_like
        Read depth of the simulation
        for each sample (row).
    dirichlet: bool
        If True Dirichlet will be
        sampled, if not Multinomial
        will be sampled directly.
    pseudocount: float
        A pseudocount for sampling the
        Dirichlet distribution. Only
        applies if dirichlet is True.
    """

    # check matrix and ensure
    # data is proportions
    mat = input_matrix_validation(mat + pseudocount,
                                  depths)
    # simulate from proportions
    if dirichlet:
        sim = np.vstack([multinomial(depths[i, 0],
                                     dirichlet(mat[i, :]))
                         for i in range(mat.shape[0])])
    else:
        sim = np.vstack([multinomial(depths[i, 0],
                                     mat[i, :])
                         for i in range(mat.shape[0])])        

    return output_matrix_validation(sim)


def input_matrix_validation(mat, depths):

    if np.any(depths < 0):
        raise ValueError("Read depth cannot have negative proportions")
    if depths.ndim != 2:
        raise ValueError("Read depth can only have two dimensions")
    if depths.shape[0] != mat.shape[0]:
        raise ValueError("Number of est. read depth does not match number of "
                         "samples in the input matrix")
    if np.all(depths <= 0):
        raise ValueError("Input matrix cannot have rows with all zeros")
    # check matrix and ensure
    # data is proportions
    mat = closure(mat)

    return mat

def output_matrix_validation(sim):

    # ensure no zero counts
    sim[sim < 0.0] = 0.0
    # remove zero sums and return a mask (if needed)
    zero_sum_mask_rows = sim.sum(1) > 0
    sim = sim[zero_sum_mask_rows]
    zero_sum_mask_columns = sim.sum(0) > 0
    sim = sim[:, zero_sum_mask_columns]

    return sim, zero_sum_mask_rows, zero_sum_mask_columns
