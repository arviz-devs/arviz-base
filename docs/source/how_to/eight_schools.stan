
data {
  int<lower=0> J;
  array[J] real y;
  array[J] real<lower=0> sigma;
}

parameters {
  real mu;
  real<lower=0> tau;
  array[J] real theta_tilde;
}

transformed parameters {
  array[J] real theta;
  for (j in 1:J)
    theta[j] = mu + tau * theta_tilde[j];
}

model {
  mu ~ normal(0, 5);
  tau ~ exponential(1);
  theta_tilde ~ normal(0, 1);
  y ~ normal(theta, sigma);
}

generated quantities {
  array[J] real log_lik;
  array[J] real y_rep;

  for (j in 1:J) {
    log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
    y_rep[j] = normal_rng(theta[j], sigma[j]);
  }
}
