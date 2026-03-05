import numpy as np


def test_posterior_predictive_mapping_cmdstanpy():
    """posterior_predictive dict mapping should rename predictive variables."""
    rng = np.random.default_rng()
    mapping = {"y": "y_hat"}
    data = {"y_hat": rng.standard_normal((1, 10))}
    renamed = {obs: data[var] for obs, var in mapping.items()}
    assert "y" in renamed
    assert "y_hat" not in renamed


def test_posterior_predictive_mapping_pystan():
    """posterior_predictive dict mapping should rename predictive variables."""
    rng = np.random.default_rng()
    mapping = {"y": "y_hat"}
    data = {"y_hat": rng.standard_normal((1, 10))}
    renamed = {obs: data[var] for obs, var in mapping.items()}
    assert "y" in renamed
    assert "y_hat" not in renamed


def test_log_likelihood_mapping_cmdstanpy():
    """log_likelihood dict mapping should rename variables."""
    rng = np.random.default_rng()
    mapping = {"y": "log_lik"}
    data = {"log_lik": rng.standard_normal((1, 10))}
    renamed = {obs: data[var] for obs, var in mapping.items()}
    assert "y" in renamed
    assert "log_lik" not in renamed


def test_log_likelihood_mapping_pystan():
    """log_likelihood dict mapping should rename variables."""
    rng = np.random.default_rng()
    mapping = {"y": "log_lik"}
    data = {"log_lik": rng.standard_normal((1, 10))}
    renamed = {obs: data[var] for obs, var in mapping.items()}
    assert "y" in renamed
    assert "log_lik" not in renamed
