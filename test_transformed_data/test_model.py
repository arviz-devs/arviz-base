"""
Minimal test for CmdStanPy transformed_data issue #147.

Contains a Stan model with transformed data block and a Python test script.
"""

import numpy as np
from cmdstanpy import CmdStanModel


def run_model():
    """Compile and sample the Stan model."""
    # Sample data
    n = 5
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data = {"N": n, "x": x}

    # Compile the Stan model
    model = CmdStanModel(stan_file="test_model.stan")

    # Sample
    fit = model.sample(data=data, chains=1, iter_sampling=10, iter_warmup=5)

    # Basic check (instead of print statements)
    assert "mu" in fit.stan_variables()


if __name__ == "__main__":
    run_model()
