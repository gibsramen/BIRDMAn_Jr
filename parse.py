import re

import pandas as pd


def read_file(filepath):
    df = pd.read_csv(filepath, skiprows=31, skipfooter=5)
    cols_to_drop = ["lp__", "accept_stat__"]
    df = df.drop(columns=cols_to_drop)

    y_sim_cols = [x for x in df.columns if "y_sim" in x]
    # read last column entry to get dimensions
    N, D = map(
        int,
        re.search("y_sim\\.(\\d+)\\.(\\d+)", y_sim_cols[-1]).groups()
    )
    y_sim = df[y_sim_cols].values.reshape(N, D, order="F")

    lam_clr_cols = [x for x in df.columns if "lam_clr" in x]
    lam_clr = df[lam_clr_cols].values.reshape(N, D, order="F")

    beta_var_cols = [x for x in df.columns if "beta_var" in x]
    p, _ = map(
        int,
        re.search("beta_var\\.(\\d+)\\.(\\d+)", beta_var_cols[-1]).groups()
    )
    beta_var = df[beta_var_cols].values.reshape(p, (D-1), order="F")

    phi_cols = [x for x in df.columns if "phi" in x]
    phi = df[phi_cols].values

    return {
        "table": df,
        "y_sim": y_sim,
        "lam_clr": lam_clr,
        "beta_var": beta_var,
        "phi": phi
    }
