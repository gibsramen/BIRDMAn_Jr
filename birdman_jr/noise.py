import numpy as np
from numpy.random import (randint, normal)
from skbio.stats.composition import (alr, alr_inv)

def add_noise(mat,
              pseudocount = 1,
              percent_normal = 0.1,
              percent_random = 0.1,
              random_count = 1,
              add_missing_at_random = False,
              percent_missing = 0.1):
    
    """
    This function transforms count data into
    the simplex with the ALR This gives the
    data an approximate normal distibution,
    allowing for the addition of normal
    and randomly distributed noise. The data
    is transformed into proportions with the 
    inverse alr and then missing values can be
    added, either to match the input or at random.

    mat: array_like
        matrix of strictly positive counts
        or probabilities/proportions.
        columns = features (components)
        rows = samples (compositions)
    pseudocount: float
        Pseudocount to add for ALR.
        Default is 1.
    percent_normal: float
        Percent of data to add homoscedastic
        noise. Default is 0.1 (i.e. 10%)
    percent_random: float
        Percent of data to add heteroscedastic
        noise. Default is 0.1 (i.e. 10%)
    random_count: float
        Intensity of random data added.
        Default is 1 (ten would be large).
    add_missing_at_random: bool
        If missing values should match the input
        mat. If True percent_missing is ignored.
    percent_missing: float
        Percent of data to add missing (zero)
        values. Default is 0.1 (i.e. 10%)    
    """

    # transform mat into ALR space for 
    # adding normal dist. noise 
    mat_noise = alr(mat + pseudocount)

    # add homo-scedastic noise
    err = pnormal * np.ones_like(mat_noise)
    mat_noise = normal(mat_noise, err)

    # add hetero-scedastic noise
    err = percent_random * np.ones_like(mat_noise)
    n_entries = int(percent_random * np.count_nonzero(mat_noise))
    i = randint(0, err.shape[0], n_entries)
    j = randint(0, err.shape[1], n_entries)
    err[i, j] = random_count
    mat_noise = normal(mat_noise, err)

    # transform back
    mat_noise = alr_inv(mat_noise)

    # finally add sparsity
    # Note: there will be no zeros after
    #       using the pseudocount
    if add_missing_at_random:
        n_entries = int(percent_missing * np.count_nonzero(mat_noise))
        i = randint(0, mat_noise.shape[0], n_entries)
        j = randint(0, mat_noise.shape[1], n_entries)
        mat_noise[i, j] = 0
    else:
        i, j = np.nonzero(mat == 0)
        mat_noise[i, j] = 0

    return mat_noise
