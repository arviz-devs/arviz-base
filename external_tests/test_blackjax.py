# pylint: disable=redefined-outer-name
"""Tests for from_blackjax converter using real BlackJAX NUTS sampling."""

from collections import namedtuple

import numpy as np
import pytest

from arviz_base import from_blackjax
from arviz_base.testing import check_multiple_attrs

from .helpers import importorskip

blackjax = importorskip("blackjax")
jax = importorskip("jax")
import jax.numpy as jnp  # noqa: PLC0415, E402

_EIGHT_SCHOOLS_Y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
_EIGHT_SCHOOLS_SIGMA = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])


def _log_prob(position):
    """Log-joint of the non-centred eight-schools model.

    Parameters are stored in a dict with keys ``mu``, ``log_tau``,
    and ``theta_tilde`` (shape 8,).
    """
    mu = position["mu"]
    log_tau = position["log_tau"]
    tau = jnp.exp(log_tau)
    theta_tilde = position["theta_tilde"]
    theta = mu + tau * theta_tilde

    # priors
    lp = 0.0
    lp += -0.5 * (mu / 5.0) ** 2  # Normal(0, 5)
    lp += log_tau - jnp.log1p((log_tau / jnp.log(5.0)) ** 2)  # Half-Cauchy(0, 5) on tau
    lp += -0.5 * jnp.sum(theta_tilde**2)  # Normal(0, 1) for each school

    # likelihood
    y = jnp.array(_EIGHT_SCHOOLS_Y)
    sigma = jnp.array(_EIGHT_SCHOOLS_SIGMA)
    lp += -0.5 * jnp.sum(((y - theta) / sigma) ** 2)

    return lp


_INIT_POSITION = {
    "mu": 0.0,
    "log_tau": 0.0,
    "theta_tilde": jnp.zeros(8),
}


def _run_nuts_single_chain(rng_key, n_warmup: int, n_draws: int):
    """Run a single-chain NUTS and return (states, infos)."""
    warmup = blackjax.window_adaptation(blackjax.nuts, _log_prob)
    (state, parameters), _ = warmup.run(rng_key, _INIT_POSITION, num_steps=n_warmup)

    kernel = blackjax.nuts(_log_prob, **parameters)

    def one_step(state, rng_key):
        state, info = kernel.step(rng_key, state)
        return state, (state, info)

    draw_keys = jax.random.split(jax.random.fold_in(rng_key, 1), n_draws)
    _, (states, infos) = jax.lax.scan(one_step, state, draw_keys)
    return states, infos


def _run_nuts_multi_chain(rng_key, n_chains: int, n_warmup: int, n_draws: int):
    """Run multi-chain NUTS via vmap and return (states, infos)."""
    chain_keys = jax.random.split(rng_key, n_chains)

    def run_chain(key):
        warmup = blackjax.window_adaptation(blackjax.nuts, _log_prob)
        (state, parameters), _ = warmup.run(key, _INIT_POSITION, num_steps=n_warmup)
        kernel = blackjax.nuts(_log_prob, **parameters)

        def one_step(state, rng_key):
            state, info = kernel.step(rng_key, state)
            return state, (state, info)

        draw_keys = jax.random.split(jax.random.fold_in(key, 1), n_draws)
        _, (states, infos) = jax.lax.scan(one_step, state, draw_keys)
        return states, infos

    states, infos = jax.vmap(run_chain)(chain_keys)
    return states, infos


N_WARMUP = 50
N_DRAWS = 100
N_CHAINS = 2


@pytest.fixture(scope="module")
def single_chain_run():
    """Real BlackJAX NUTS single-chain run.  Returns (states, infos)."""
    rng_key = jax.random.PRNGKey(0)
    return _run_nuts_single_chain(rng_key, N_WARMUP, N_DRAWS)


@pytest.fixture(scope="module")
def multi_chain_run():
    """Real BlackJAX NUTS multi-chain run.  Returns (states, infos)."""
    rng_key = jax.random.PRNGKey(1)
    return _run_nuts_multi_chain(rng_key, N_CHAINS, N_WARMUP, N_DRAWS)


class TestSingleChain:
    """Tests using a real single-chain NUTS run.

    Single-chain output has shape (n_draws, *event_shape) with no leading chain
    axis, so all tests must pass sample_dims=["draw"] explicitly.
    """

    def test_posterior_and_no_sample_stats(self, single_chain_run):
        """Posterior present; sample_stats absent when info is not passed."""
        states, _ = single_chain_run
        idata = from_blackjax(posterior=states.position, sample_dims=["draw"])
        fails = check_multiple_attrs(
            {
                "posterior": ["mu", "log_tau", "theta_tilde"],
                "~sample_stats": [],
            },
            idata,
        )
        assert not fails
        assert "chain" not in idata.posterior.dims
        assert idata.posterior.sizes["draw"] == N_DRAWS

    def test_with_sample_stats(self, single_chain_run):
        """sample_stats are populated from NUTS info object."""
        states, infos = single_chain_run
        idata = from_blackjax(posterior=states.position, info=infos, sample_dims=["draw"])
        fails = check_multiple_attrs(
            {
                "posterior": ["mu", "log_tau", "theta_tilde"],
                "sample_stats": ["acceptance_rate", "diverging", "energy", "tree_depth"],
            },
            idata,
        )
        assert not fails
        assert idata.sample_stats.sizes["draw"] == N_DRAWS

    def test_reached_max_tree_depth(self, single_chain_run):
        """reached_max_tree_depth is added when max_tree_depth is provided."""
        states, infos = single_chain_run
        idata = from_blackjax(
            posterior=states.position,
            info=infos,
            max_tree_depth=10,
            sample_dims=["draw"],
        )
        fails = check_multiple_attrs(
            {"sample_stats": ["tree_depth", "reached_max_tree_depth"]},
            idata,
        )
        assert not fails

    def test_named_tuple_position(self, single_chain_run):
        """NamedTuple positions are parsed correctly."""
        states, _ = single_chain_run
        pos = states.position
        Position = namedtuple("Position", pos.keys())
        nt_position = Position(**pos)
        idata = from_blackjax(posterior=nt_position, sample_dims=["draw"])
        fails = check_multiple_attrs({"posterior": list(pos.keys())}, idata)
        assert not fails

    def test_bare_array_position(self):
        """Bare JAX array positions are stored under the key 'x'."""
        arr = jnp.ones((N_DRAWS,))
        idata = from_blackjax(posterior=arr, sample_dims=["draw"])
        fails = check_multiple_attrs({"posterior": ["x"]}, idata)
        assert not fails

    def test_jax_arrays_converted_to_numpy(self, single_chain_run):
        """JAX arrays in position are converted to numpy in the output."""
        states, _ = single_chain_run
        idata = from_blackjax(posterior=states.position, sample_dims=["draw"])
        assert isinstance(idata.posterior["mu"].values, np.ndarray)

    def test_coords_and_dims(self, single_chain_run):
        """Coordinates and dims are applied correctly."""
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
        states, _ = single_chain_run
        idata = from_blackjax(
            posterior=states.position,
            coords={"school": school},
            dims={"theta_tilde": ["school"]},
            sample_dims=["draw"],
        )
        assert "school" in idata.posterior.coords
        assert list(idata.posterior.coords["school"].values) == school

    def test_posterior_predictive(self, single_chain_run):
        """posterior_predictive group is created when provided."""
        states, _ = single_chain_run
        rng = np.random.default_rng(5)
        pp = {"y_hat": rng.normal(size=(N_DRAWS, 8))}
        idata = from_blackjax(
            posterior=states.position, posterior_predictive=pp, sample_dims=["draw"]
        )
        fails = check_multiple_attrs({"posterior_predictive": ["y_hat"]}, idata)
        assert not fails

    def test_observed_data(self, single_chain_run):
        """observed_data has no sample dims."""
        states, _ = single_chain_run
        obs = {"y": _EIGHT_SCHOOLS_Y}
        idata = from_blackjax(posterior=states.position, observed_data=obs, sample_dims=["draw"])
        fails = check_multiple_attrs({"observed_data": ["y"]}, idata)
        assert not fails
        assert "draw" not in idata.observed_data.sizes

    def test_constant_data(self, single_chain_run):
        """constant_data has no sample dims."""
        states, _ = single_chain_run
        cdata = {"sigma": _EIGHT_SCHOOLS_SIGMA}
        idata = from_blackjax(posterior=states.position, constant_data=cdata, sample_dims=["draw"])
        fails = check_multiple_attrs({"constant_data": ["sigma"]}, idata)
        assert not fails
        assert "draw" not in idata.constant_data.sizes

    def test_prior_without_posterior(self):
        """When no posterior is given, all prior vars go into the prior group."""
        rng = np.random.default_rng(9)
        prior = {
            "mu": rng.normal(size=(N_DRAWS,)),
            "log_tau": rng.normal(size=(N_DRAWS,)),
        }
        idata = from_blackjax(prior=prior, sample_dims=["draw"])
        fails = check_multiple_attrs(
            {"prior": ["mu", "log_tau"], "~prior_predictive": []},
            idata,
        )
        assert not fails

    def test_prior_splits_with_posterior(self, single_chain_run):
        """Vars in posterior go to prior; extra vars go to prior_predictive."""
        states, _ = single_chain_run
        rng = np.random.default_rng(11)
        prior = {
            "mu": rng.normal(size=(N_DRAWS,)),
            "log_tau": rng.normal(size=(N_DRAWS,)),
            "y_hat": rng.normal(size=(N_DRAWS, 8)),
        }
        idata = from_blackjax(posterior=states.position, prior=prior, sample_dims=["draw"])
        fails = check_multiple_attrs(
            {
                "prior": ["mu", "log_tau"],
                "prior_predictive": ["y_hat"],
            },
            idata,
        )
        assert not fails

    def test_no_prior(self, single_chain_run):
        """prior and prior_predictive groups absent when no prior is provided."""
        states, _ = single_chain_run
        idata = from_blackjax(posterior=states.position, sample_dims=["draw"])
        fails = check_multiple_attrs({"~prior": [], "~prior_predictive": []}, idata)
        assert not fails


class TestMultiChain:
    """Tests using a real multi-chain NUTS run via jax.vmap.

    Multi-chain output has shape (n_chains, n_draws, *event_shape) so the
    default sample_dims=["chain", "draw"] works out of the box with no
    extra arguments needed.
    """

    def test_posterior_and_sample_stats(self, multi_chain_run):
        """Multi-chain run works out of the box with default sample_dims."""
        states, infos = multi_chain_run
        idata = from_blackjax(
            posterior=states.position,
            info=infos,
        )
        fails = check_multiple_attrs(
            {
                "posterior": ["mu", "log_tau", "theta_tilde"],
                "sample_stats": ["acceptance_rate", "diverging", "tree_depth"],
            },
            idata,
        )
        assert not fails
        assert idata.posterior.sizes["chain"] == N_CHAINS
        assert idata.posterior.sizes["draw"] == N_DRAWS
        assert idata.sample_stats.sizes["chain"] == N_CHAINS
        assert idata.sample_stats.sizes["draw"] == N_DRAWS

    def test_reached_max_tree_depth(self, multi_chain_run):
        """reached_max_tree_depth added in multi-chain run."""
        states, infos = multi_chain_run
        idata = from_blackjax(
            posterior=states.position,
            info=infos,
            max_tree_depth=10,
        )
        fails = check_multiple_attrs(
            {"sample_stats": ["tree_depth", "reached_max_tree_depth"]},
            idata,
        )
        assert not fails


class TestSampleDims:
    """Tests for the sample_dims parameter."""

    def test_default_sample_dims_gives_chain_draw(self, multi_chain_run):
        """Default sample_dims=["chain", "draw"] works with multi-chain output."""
        states, _ = multi_chain_run
        idata = from_blackjax(posterior=states.position)
        assert "chain" in idata.posterior.dims
        assert "draw" in idata.posterior.dims
        assert idata.posterior.sizes["chain"] == N_CHAINS
        assert idata.posterior.sizes["draw"] == N_DRAWS

    def test_sample_dims_draw_for_single_chain(self, single_chain_run):
        """sample_dims=["draw"] correctly handles single-chain output."""
        states, _ = single_chain_run
        idata = from_blackjax(posterior=states.position, sample_dims=["draw"])
        assert "draw" in idata.posterior.dims
        assert "chain" not in idata.posterior.dims
        assert idata.posterior.sizes["draw"] == N_DRAWS

    def test_sample_dims_sample_no_chain(self, single_chain_run):
        """sample_dims=['sample'] stores flat samples without chain or draw dim."""
        states, _ = single_chain_run
        idata = from_blackjax(posterior=states.position, sample_dims=["sample"])
        assert "sample" in idata.posterior.dims
        assert "chain" not in idata.posterior.dims
        assert "draw" not in idata.posterior.dims
        assert idata.posterior.sizes["sample"] == N_DRAWS

    def test_sample_dims_sample_with_info(self, single_chain_run):
        """sample_dims=['sample'] also applies to sample_stats."""
        states, infos = single_chain_run
        idata = from_blackjax(
            posterior=states.position,
            info=infos,
            sample_dims=["sample"],
        )
        fails = check_multiple_attrs(
            {"sample_stats": ["acceptance_rate", "diverging"]},
            idata,
        )
        assert not fails
        assert "chain" not in idata.sample_stats.dims
        assert idata.sample_stats.sizes["sample"] == N_DRAWS
