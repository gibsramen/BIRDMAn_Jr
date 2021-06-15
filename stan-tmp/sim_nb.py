import cmdstanpy
import numpy as np

m = cmdstanpy.CmdStanModel(stan_file="sim_nb.stan")

rng = np.random.default_rng()

N = 50
D = 20
depth = np.log(rng.poisson(100, size=N))
B_p = 1
phi_s = 1
x = np.ones([N, 2])
x[0:int(N/2), 1] = 0

data = {
    "N": N,
    "D": D,
    "depth": depth,
    "x": x,
    "B_p": B_p,
    "phi_s": phi_s,
}

m.sample(
    fixed_param=True,
    data=data,
    output_dir="output",
    iter_sampling=1,
)
