
data {
    int<lower=0> N;
    vector[N] x;
}

transformed data {
    vector[N] x_sq;
    for (i in 1:N)
        x_sq[i] = x[i] * x[i];
}

parameters {
    real mu;
}

model {
    mu ~ normal(0, 1);
}
