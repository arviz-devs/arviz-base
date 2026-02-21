from cmdstanpy import CmdStanModel
import numpy as np

# Sample data
N = 5
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
data = {'N': N, 'x': x}

# Compile the Stan model
model = CmdStanModel(stan_file='test_model.stan')

# Sample
fit = model.sample(data=data, chains=1, iter_sampling=10, iter_warmup=5)

# Inspect outputs
print("stan_variables:", fit.stan_variables())
print("column_names:", fit.column_names)
