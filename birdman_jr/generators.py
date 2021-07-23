import numpy as np
from scipy.stats import norm


def gradient(g, mu, sigma, peaks=None):
    """
    This generates an urn simulating a chain of interacting species.
    This commonly occurs in the context of a redox tower, where
    multiple species are distributed across a gradient.
    Parameters
    ----------
    g: array_like
       Vector of values associated with an underlying gradient.
    mu: array_like
       Vector of means.
    sigma: array_like
       Vector of standard deviations.
    peaks : array_like
       Vector of peaks
    rng: np.random.RandomState
       Numpy random state.
    Returns
    -------
    np.array
       A matrix of real-valued positive abundances where
       there are `n` rows and `m` columns where `n` corresponds
       to the number of samples along the `gradient` and `m`
       corresponds to the number of species in `mus`.
    """
    if peaks is None:
        peaks = np.ones(len(mu))

    xs = [peaks[i] * norm.pdf(g, loc=mu[i], scale=sigma[i])
          for i in range(len(mu))]
    return np.vstack(xs).T

def blocks(ncols, nrows, nblocks, overlap=0, minval=0, sigma=2, maxval=1.0):
    """
    Generate block diagonal with Gaussian distributed values within blocks.

    Parameters
    ----------

    ncol : int
        Number of columns

    nrows : int
        Number of rows

    nblocks : int
        Number of blocks, mucst be greater than one

    overlap : int
        The Number of overlapping columns (Default = 0)

    minval : int
        The min value output of the table (Default = 0)

    maxval : int
        The max value output of the table (Default = 1)


    Returns
    -------
    np.array
        Table with a block diagonal where the rows represent samples
        and the columns represent features.  The values within the blocks
        are gaussian distributed between 0 and 1.
    Note
    ----
    The number of blocks specified by `nblocks` needs to be greater than 1.

    """

    if nblocks <= 1:
        raise ValueError('`nblocks` needs to be greater than 1.')
    mat = np.zeros((nrows, ncols))
    gradient = np.linspace(0, 10, nrows)
    mu = np.linspace(0, 10, ncols)
    xs = [norm.pdf(gradient, loc=mu[i], scale=sigma)
          for i in range(len(mu))]
    mat = np.vstack(xs).T
    block_cols = ncols // (nblocks * 2)
    block_rows = nrows // nblocks 

    for b in range(nblocks - 1):

        gradient = np.linspace(
            5, 5, block_rows)  # samples (bock_rows)
        # features (block_cols+overlap)
        mu = np.linspace(0, 10, block_cols + overlap)
        xs = [norm.pdf(gradient, loc=mu[i], scale=sigma)
              for i in range(len(mu))]

        B = np.vstack(xs).T * maxval
        lower_row = block_rows * b
        upper_row = min(block_rows * (b + 1), nrows)
        lower_col = block_cols * b
        upper_col = min(block_cols * (b + 1), ncols)

        if b == 0:
            mat[lower_row:upper_row,
                lower_col:int(upper_col + overlap)] = B
        else:
            ov_tmp = int(overlap / 2)
            if (B.shape) == (mat[lower_row:upper_row,
                                 int(lower_col - ov_tmp):
                                 int(upper_col + ov_tmp + 1)].shape):
                mat[lower_row:upper_row, int(
                    lower_col - ov_tmp):int(upper_col + ov_tmp + 1)] = B
            elif (B.shape) == (mat[lower_row:upper_row,
                                   int(lower_col - ov_tmp):
                                   int(upper_col + ov_tmp)].shape):
                mat[lower_row:upper_row, int(
                    lower_col - ov_tmp):int(upper_col + ov_tmp)] = B
            elif (B.shape) == (mat[lower_row:upper_row,
                                   int(lower_col - ov_tmp):
                                   int(upper_col + ov_tmp - 1)].shape):
                mat[lower_row:upper_row, int(
                    lower_col - ov_tmp):int(upper_col + ov_tmp - 1)] = B

    upper_col = int(upper_col - overlap) 
    # Make last block fill in the remainder
    gradient = np.linspace(5, 5, nrows - upper_row)
    mu = np.linspace(0, 10, ncols - upper_col)
    xs = [norm.pdf(gradient, loc=mu[i], scale=sigma)
          for i in range(len(mu))]
    B = np.vstack(xs).T * maxval

    mat[upper_row:, upper_col:] = B

    return mat
