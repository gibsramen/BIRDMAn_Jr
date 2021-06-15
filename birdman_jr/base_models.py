import numpy as np
from skbio.stats.composition import closure
from numpy.random import (poisson, lognormal, gamma,
                          dirichlet, multinomial)


def poisson_lognormal(mat, depths, kappa=1):

    """
    Simulate from counts, probabilities, or
    proportions of input matrix with a
    Poisson Log-Normal distribution.

    Parameters
    ----------
    mat: array_like
        matrix of strictly positive counts
        or probabilities/proportions.
        columns = features (components)
        rows = samples (compositions)
    depth : array_like
        Read depth of the simulation
        for each sample (row).
    kappa: float
        Over-dispersion parameter.
        Default is 1.

    Returns
    -------
    array_like, np.int
       A matrix of counts simulated from
       the input mat by the distribution.
    list, bool
        Mask of rows that summed to zero
    list, bool
        Mask of columns that summed to zero

    Raises
    ------
    ValueError
       Raises an error if any depths are equal
       or less than zero.
    ValueError
       Raises an error if any depths does not have
       exactly 2 dimensions.
    ValueError
       Raises an error if any depths shape does not match the
       input matrix.
    ValueError
       Raises an error if any values are negative.
    ValueError
       Raises an error if the matrix has more than 2 dimension.
    ValueError
       Raises an error if there is a row that has all zeros.

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

    Parameters
    ----------
    mat: array_like
        matrix of strictly positive counts
        or probabilities/proportions.
        columns = features (components)
        rows = samples (compositions)
    depth : array_like
        Read depth of the simulation
        for each sample (row).
    kappa: float
        Over-dispersion parameter.
        Default is 1.

    Returns
    -------
    array_like, np.int
       A matrix of counts simulated from
       the input mat by the distribution.
    list, bool
        Mask of rows that summed to zero
    list, bool
        Mask of columns that summed to zero

    Raises
    ------
    ValueError
       Raises an error if any depths are equal
       or less than zero.
    ValueError
       Raises an error if any depths does not have
       exactly 2 dimensions.
    ValueError
       Raises an error if any depths shape does not match the
       input matrix.
    ValueError
       Raises an error if any values are negative.
    ValueError
       Raises an error if the matrix has more than 2 dimension.
    ValueError
       Raises an error if there is a row that has all zeros.

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
                          use_dirichlet=False,
                          pseudocount=0.001):

    """
    Simulate from counts, probabilities, or
    proportions of input matrix with a
    Dirichlet Multinomial distribution.

    Parameters
    ----------
    mat: array_like
        matrix of strictly positive counts
        or probabilities/proportions.
        columns = features (components)
        rows = samples (compositions)
    depth : array_like
        Read depth of the simulation
        for each sample (row).
    use_dirichlet: bool
        If True Dirichlet will be
        sampled, if not Multinomial
        will be sampled directly.
    pseudocount: float
        A pseudocount for sampling the
        Dirichlet distribution. Only
        applies if dirichlet is True.

    Returns
    -------
    array_like, np.int
       A matrix of counts simulated from
       the input mat by the distribution.
    list, bool
        Mask of rows that summed to zero
    list, bool
        Mask of columns that summed to zero

    Raises
    ------
    ValueError
       Raises an error if any depths are equal
       or less than zero.
    ValueError
       Raises an error if any depths does not have
       exactly 2 dimensions.
    ValueError
       Raises an error if any depths shape does not match the
       input matrix.
    ValueError
       Raises an error if any values are negative.
    ValueError
       Raises an error if the matrix has more than 2 dimension.
    ValueError
       Raises an error if there is a row that has all zeros.

    """

    # with or w/o dirichlet
    if use_dirichlet:
        # check matrix and ensure
        # data is proportions
        mat = input_matrix_validation(mat + pseudocount,
                                      depths)
        sim = np.vstack([multinomial(depths[i, 0],
                                     dirichlet(mat[i, :]))
                         for i in range(mat.shape[0])])
    else:
        # check matrix and ensure
        # data is proportions
        mat = input_matrix_validation(mat, depths)
        sim = np.vstack([multinomial(depths[i, 0],
                                     mat[i, :])
                         for i in range(mat.shape[0])])

    return output_matrix_validation(sim)


def input_matrix_validation(mat, depths):

    if np.any(depths <= 0):
        raise ValueError("Read depth cannot have values "
                         "less than or equal to zero")
    if depths.ndim != 2:
        raise ValueError("Read depth can only have two dimensions")
    if depths.shape[0] != mat.shape[0]:
        raise ValueError("Number of est. read depth does not match number of "
                         "samples in the input matrix")
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
