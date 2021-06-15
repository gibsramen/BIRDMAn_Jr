from biom import Table
from birdman_jr.noise import add_noise
from birdman_jr.base_models import (poisson_lognormal,
                                    dirichlet_multinomial,
                                    negative_binomial)


def simulate(table,
             depths=None,
             distribution="pln",
             kappa=1,
             pseudocount=1,
             impose_noise=False,
             percent_normal=0.1,
             percent_random=0.1,
             random_count=1,
             add_missing_at_random=False,
             percent_missing=0.1):
    """
    This function will take and input table
    and simulate on the proportions of the data
    itself. The benefit is the data can be altered
    to have smaller/larger depth variance or over-
    dispersion (i.e. kappa). Additionally, noise
    can be added both normally and randomly. Sparsity
    can also be increased or retained to match the
    original data.

    Parameters
    ----------
    table: biom.Table
        Feature table (features x samples)
    depths: array_like or None
        The depth of each sample
        if depth is None then the
        depth input table depth
        will be used.
        Default is None.
    distribution: str
        The type of distribution to
        use. Options are:
        Poisson Log-Normal (or pln),
        Negative Binomial (or nb),
        Dirichlet Multinomial (or dm), or
        Multinomial (or m).
        Default is Poisson Log-Normal/pln.
    kappa: float
        Over-dispersion parameter.
        Only applies for pln and nb.
        Default is 1.
    pseudocount: float
        Pseudocount to add for ALR
        and/or Dirichlet.
        Default is 1.
    impose_noise: bool
        If to add noise to the data.
        If False no noise will be added
        to the data. Thus percent_normal,
        percent_random, add_missing_at_random,
        and percent_missing would be ignored.
        Default is False.
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

    Returns
    -------
    biom.Table
       A table of the simulated data on the
       input data based on distribution chosen.

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

    # check model name is correct
    allowed_dists = ["Poisson Log-Normal", "pln",
                     "Negative Binomial", "nb",
                     "Dirichlet Multinomial", "dm",
                     "Multinomial", "m"]
    if distribution not in allowed_dists:
        allow_str = ", ".join(allowed_dists)
        ValueError("distribution must be one of %s" % allow_str)
    # get data as table
    mat = table.matrix_data.toarray().T
    # get depths if not provided
    if depths is None:
        depths = mat.sum(1).reshape(mat.shape[0], -1)
    # add noise, if requested
    if impose_noise:
        mat = add_noise(mat, pseudocount, percent_normal,
                        percent_random, random_count,
                        add_missing_at_random, percent_missing)
    # run model simulation and return
    if distribution in ["Poisson Log-Normal", "pln"]:
        sim_res = poisson_lognormal(mat, depths, kappa=kappa)
    elif distribution in ["Negative Binomial", "nb"]:
        sim_res = negative_binomial(mat, depths, kappa=kappa)
    elif distribution in ["Dirichlet Multinomial", "dm"]:
        sim_res = dirichlet_multinomial(mat, depths,
                                        use_dirichlet=True,
                                        pseudocount=pseudocount)
    elif distribution in ["Multinomial", "m"]:
        sim_res = dirichlet_multinomial(mat, depths)

    # make table to return
    simulation_table = Table(sim_res[0],
                             table.ids("observation")[sim_res[2]],
                             table.ids()[sim_res[1]])

    return simulation_table
