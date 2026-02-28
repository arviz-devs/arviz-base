(cmdstanpy_conversion)=
# Converting CmdStanPy objects to DataTree

ArviZ offers the {func}`~arviz_base.from_cmdstanpy` function to convert CmdStanPy results into
`DataTree`, the data structure used by ArviZ.

```python
import arviz_base as az
from cmdstanpy import CmdStanModel

model = CmdStanModel(stan_file="bernoulli.stan")
fit = model.sample({"N": 10, "y": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]})

idata = az.from_cmdstanpy(posterior=fit)
idata
```
