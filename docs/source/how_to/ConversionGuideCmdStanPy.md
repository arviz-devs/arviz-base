# Conversion Guide: CmdStanPy

This example demonstrates the recommended way to include transformed data when converting CmdStanPy results to ArviZ.

CmdStanPy does not expose variables from the `transformed data` block in the sampling output. Instead, users should compute the transformed variables in Python and pass them via the `constant_data` argument.

## Example

```python
import numpy as np
import arviz as az
from cmdstanpy import CmdStanModel

# Original data
x = np.array([1.0, 2.0, 3.0, 4.0])
data = {"N": len(x), "x": x}

# Compute transformed data in Python
logx = np.log(x)

# Compile and sample model
model = CmdStanModel(stan_file="model.stan")
fit = model.sample(data=data)

# Convert to ArviZ and include transformed data
idata = az.from_cmdstanpy(
    posterior=fit,
    constant_data={"x": x, "logx": logx},
)

print(idata)
```

This ensures transformed variables are available in the `constant_data` group without requiring duplication in Stan generated quantities.
