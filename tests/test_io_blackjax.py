# pylint: disable=redefined-outer-name
from collections import namedtuple

import numpy as np
import pytest

from arviz_base import from_blackjax
from arviz_base.testing import check_multiple_attrs


@pytest.fixture(scope="module")
def draws():
    return 10


@pytest.fixture(scope="module")
def chains():
    return 3


@pytest.fixture(scope="function")
def single_chain_position(draws):
    rng = np.random.default_rng(42)
    return {
        "mu": rng.normal(size=(draws,)),
        "tau": rng.normal(size=(draws,)),
        "theta": rng.normal(size=(draws, 8)),
    }


@pytest.fixture(scope="function")
def multi_chain_position(draws, chains):
    rng = np.random.default_rng(42)
    return {
        "mu": rng.normal(size=(chains, draws)),
        "tau": rng.normal(size=(chains, draws)),
        "theta": rng.normal(size=(chains, draws, 8)),
    }


@pytest.fixture(scope="function")
def nuts_info_single(draws):
    rng = np.random.default_rng(0)
    return {
        "num_integration_steps": rng.integers(1, 8, size=(draws,)),
        "acceptance_rate": rng.uniform(size=(draws,)),
        "is_divergent": rng.choice([True, False], size=(draws,), p=[0.05, 0.95]),
        "energy": rng.normal(size=(draws,)),
        "step_size": np.full((draws,), 0.01),
    }


@pytest.fixture(scope="function")
def nuts_info_multi(draws, chains):
    rng = np.random.default_rng(0)
    return {
        "num_integration_steps": rng.integers(1, 8, size=(chains, draws)),
        "acceptance_rate": rng.uniform(size=(chains, draws)),
        "is_divergent": rng.choice([True, False], size=(chains, draws), p=[0.05, 0.95]),
        "energy": rng.normal(size=(chains, draws)),
        "step_size": np.full((chains, draws), 0.01),
    }


def test_from_blackjax_single_chain(single_chain_position, draws):
    # single-chain position should get a chain dim of 1 inserted automatically
    idata = from_blackjax(posterior=single_chain_position)
    fails = check_multiple_attrs({"posterior": ["mu", "tau", "theta"]}, idata)
    assert not fails
    assert idata.posterior.sizes["chain"] == 1
    assert idata.posterior.sizes["draw"] == draws


def test_from_blackjax_multi_chain(multi_chain_position, draws, chains):
    idata = from_blackjax(posterior=multi_chain_position, num_chains=chains)
    fails = check_multiple_attrs({"posterior": ["mu", "tau", "theta"]}, idata)
    assert not fails
    assert idata.posterior.sizes["chain"] == chains
    assert idata.posterior.sizes["draw"] == draws


def test_from_blackjax_named_tuple_position(draws):
    rng = np.random.default_rng(7)
    Position = namedtuple("Position", ["mu", "tau"])
    position = Position(mu=rng.normal(size=(draws,)), tau=rng.normal(size=(draws,)))

    idata = from_blackjax(posterior=position)
    fails = check_multiple_attrs({"posterior": ["mu", "tau"]}, idata)
    assert not fails
    assert idata.posterior.sizes["chain"] == 1
    assert idata.posterior.sizes["draw"] == draws


def test_from_blackjax_bare_array_position(draws):
    # bare array position is stored under the variable name "x"
    rng = np.random.default_rng(3)
    idata = from_blackjax(posterior=rng.normal(size=(draws,)))
    fails = check_multiple_attrs({"posterior": ["x"]}, idata)
    assert not fails


def test_from_blackjax_nuts_info_single_chain(single_chain_position, nuts_info_single, draws):
    idata = from_blackjax(posterior=single_chain_position, info=nuts_info_single)
    fails = check_multiple_attrs(
        {"sample_stats": ["n_steps", "acceptance_rate", "diverging", "energy", "tree_depth"]},
        idata,
    )
    assert not fails
    assert idata.sample_stats.sizes["chain"] == 1
    assert idata.sample_stats.sizes["draw"] == draws


def test_from_blackjax_nuts_info_multi_chain(multi_chain_position, nuts_info_multi, draws, chains):
    idata = from_blackjax(posterior=multi_chain_position, info=nuts_info_multi, num_chains=chains)
    fails = check_multiple_attrs(
        {"sample_stats": ["n_steps", "acceptance_rate", "diverging", "tree_depth"]},
        idata,
    )
    assert not fails
    assert idata.sample_stats.sizes["chain"] == chains
    assert idata.sample_stats.sizes["draw"] == draws


def test_from_blackjax_no_info(single_chain_position):
    # sample_stats group should be absent when no info is provided
    idata = from_blackjax(posterior=single_chain_position)
    assert "sample_stats" not in idata


def test_from_blackjax_jax_arrays(draws):
    pytest.importorskip("jax")
    import jax.numpy as jnp  # noqa: PLC0415

    position = {"mu": jnp.ones((draws,)), "tau": jnp.zeros((draws,))}
    idata = from_blackjax(posterior=position)

    assert isinstance(idata.posterior["mu"].values, np.ndarray)
    np.testing.assert_array_equal(idata.posterior["mu"].values[0], np.ones(draws))


def test_from_blackjax_posterior_predictive(single_chain_position, draws):
    rng = np.random.default_rng(5)
    pp = {"y_hat": rng.normal(size=(1, draws, 8))}
    idata = from_blackjax(posterior=single_chain_position, posterior_predictive=pp)
    fails = check_multiple_attrs({"posterior_predictive": ["y_hat"]}, idata)
    assert not fails


def test_from_blackjax_observed_data(single_chain_position):
    obs = {"y": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])}
    idata = from_blackjax(posterior=single_chain_position, observed_data=obs)
    fails = check_multiple_attrs({"observed_data": ["y"]}, idata)
    assert not fails
    # observed_data must not have chain/draw dims
    assert "chain" not in idata.observed_data.sizes
    assert "draw" not in idata.observed_data.sizes


def test_from_blackjax_constant_data(single_chain_position):
    cdata = {"sigma": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])}
    idata = from_blackjax(posterior=single_chain_position, constant_data=cdata)
    fails = check_multiple_attrs({"constant_data": ["sigma"]}, idata)
    assert not fails
    assert "chain" not in idata.constant_data.sizes
    assert "draw" not in idata.constant_data.sizes


def test_from_blackjax_coords_and_dims(single_chain_position):
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
    idata = from_blackjax(
        posterior=single_chain_position,
        coords={"school": school},
        dims={"theta": ["school"]},
    )
    assert "school" in idata.posterior.coords
    assert list(idata.posterior.coords["school"].values) == school


def test_from_blackjax_prior_without_posterior(draws):
    # when no posterior is given, all prior vars go into the prior group
    rng = np.random.default_rng(9)
    prior = {
        "mu": rng.normal(size=(1, draws)),
        "tau": rng.normal(size=(1, draws)),
    }
    idata = from_blackjax(prior=prior)
    fails = check_multiple_attrs({"prior": ["mu", "tau"]}, idata)
    assert not fails
    assert "prior_predictive" not in idata


def test_from_blackjax_prior_splits_with_posterior(draws):
    # vars in posterior go to prior; vars not in posterior go to prior_predictive
    rng = np.random.default_rng(11)
    position = {"mu": rng.normal(size=(draws,)), "tau": rng.normal(size=(draws,))}
    prior = {
        "mu": rng.normal(size=(1, draws)),
        "tau": rng.normal(size=(1, draws)),
        "y_hat": rng.normal(size=(1, draws, 8)),
    }
    idata = from_blackjax(posterior=position, prior=prior)
    fails = check_multiple_attrs({"prior": ["mu", "tau"], "prior_predictive": ["y_hat"]}, idata)
    assert not fails


def test_from_blackjax_no_prior(single_chain_position):
    # prior and prior_predictive groups are absent when no prior is provided
    idata = from_blackjax(posterior=single_chain_position)
    assert "prior" not in idata
    assert "prior_predictive" not in idata
