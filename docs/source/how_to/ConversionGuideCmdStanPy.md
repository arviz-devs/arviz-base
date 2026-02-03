# CmdStanPy Conversion Guide

ArviZ can convert CmdStanPy sampling results into the ArviZ data structure using `from_cmdstanpy`.

```python
import arviz_base as az
from cmdstanpy import CmdStanModel

model = CmdStanModel(stan_file="bernoulli.stan")
fit = model.sample({"N": 10, "y": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]})

idata = az.from_cmdstanpy(posterior=fit)
az.plot_trace_dist(idata)
