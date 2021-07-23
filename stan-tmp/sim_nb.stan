data {
  int<lower=0> N;
  int<lower=0> D;
  real depth[N];
  matrix[N, 2] x;
  real<lower=0> B_p;
  real<lower=0> phi_s;
}

model {

}

generated quantities {
  matrix[N, D] y_sim;
  matrix[2, D-1] beta_var;
  matrix[N, D] lam_clr;
  vector[D] phi;

  # first setup beta variables
  # inefficient but runs in like a second so whatever
  for (n in 1:N) {
    for (i in 1:D-1) {
      vector[2] tmp;
      tmp[1] = normal_rng(-2, B_p);
      if (x[n, 2] == 1)
        tmp[2] = normal_rng(0, B_p);
      else
        tmp[2] = normal_rng(1, B_p);
      beta_var[, i] = tmp;
    }
  }

  lam_clr = append_col(to_vector(rep_array(0, N)), x*beta_var);

  for (n in 1:N) {
    for (i in 1:D) {
      # for some reason restricting cauchy to > 0 made everything break?
      phi[i] = abs(cauchy_rng(0, phi_s));
      y_sim[n, i] = neg_binomial_2_log_rng(depth[n] + lam_clr[n, i], phi[i]);
    }
  }
}
