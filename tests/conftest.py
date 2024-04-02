# pylint: disable=redefined-outer-name
"""Test configuration and global fixtures."""
import numpy as np
import pytest
from arviz_base import from_dict


@pytest.fixture(scope="module")
def draws():
    """Share default draw count."""
    return 10


@pytest.fixture(scope="module")
def chains():
    """Share default chain count."""
    return 3


@pytest.fixture(scope="module")
def centered_eight(draws, chains):
    """Share default chain count."""
    rng = np.random.default_rng(31)
    mu = rng.normal(size=(chains, draws))
    theta = rng.normal(size=(chains, draws, 8))
    tau = rng.normal(size=(chains, draws))
    diverging = rng.choice([True, False], size=(chains, draws), p=[0.1, 0.9])
    mu_prior = rng.normal(size=(chains, draws))
    theta_prior = rng.normal(size=(chains, draws, 8))
    tau_prior = rng.normal(size=(chains, draws))
    y = rng.normal(size=(chains, draws, 8))
    school = [
        "Choate",
        "Deerfield",
        "Phillips Andover",
        "Phillips Exeter",
        "Hotchkiss",
        "Lawrenceville",
        "St. Paul's",
        "Mt. Hermon",
    ]

    return from_dict(
        {
            "posterior": {"mu": mu, "theta": theta, "tau": tau},
            "sample_stats": {"diverging": diverging},
            "prior": {"mu": mu_prior, "theta": theta_prior, "tau": tau_prior},
            "posterior_predictive": {"y": y},
        },
        dims={"theta": ["school"], "y": ["school"]},
        coords={"school": school},
    )


@pytest.fixture(scope="module")
def eight_schools_params():
    """Share setup for eight schools."""
    return {
        "J": 8,
        "y": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
        "sigma": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
    }
