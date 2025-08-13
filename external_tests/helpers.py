# pylint: disable=redefined-outer-name, comparison-with-callable, protected-access
"""Test helper functions for external tests."""

import gzip
import importlib
import logging
import os
import sys
from typing import Any

import cloudpickle
import numpy as np
import pytest
from _pytest.outcomes import Skipped
from packaging.version import Version

_log = logging.getLogger(__name__)


def _emcee_lnprior(theta):
    """Proper function to allow pickling."""
    mu, tau, eta = theta[0], theta[1], theta[2:]
    # Half-cauchy prior, hwhm=25
    if tau < 0:
        return -np.inf
    prior_tau = -np.log(tau**2 + 25**2)
    prior_mu = -((mu / 10) ** 2)  # normal prior, loc=0, scale=10
    prior_eta = -np.sum(eta**2)  # normal prior, loc=0, scale=1
    return prior_mu + prior_tau + prior_eta


def _emcee_lnprob(theta, y, sigma):
    """Proper function to allow pickling."""
    mu, tau, eta = theta[0], theta[1], theta[2:]
    prior = _emcee_lnprior(theta)
    like_vect = -(((mu + tau * eta - y) / sigma) ** 2)
    like = np.sum(like_vect)
    rng = np.random.default_rng()
    return like + prior, (like_vect, rng.normal((mu + tau * eta), sigma))


def emcee_schools_model(data, draws, chains):
    """Schools model in emcee."""
    import emcee

    chains = 10 * chains  # emcee is sad with too few walkers
    y = data["y"]
    sigma = data["sigma"]
    J = data["J"]  # pylint: disable=invalid-name
    ndim = J + 2

    rng = np.random.default_rng()
    pos = rng.normal(size=(chains, ndim))
    pos[:, 1] = np.absolute(pos[:, 1])  #  pylint: disable=unsupported-assignment-operation

    here = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(here, "saved_models")
    filepath = os.path.join(data_directory, "reader_testfile.h5")
    backend = emcee.backends.HDFBackend(filepath)  # pylint: disable=no-member
    backend.reset(chains, ndim)
    # pylint: disable=unexpected-keyword-arg
    sampler = emcee.EnsembleSampler(chains, ndim, _emcee_lnprob, args=(y, sigma), backend=backend)
    # pylint: enable=unexpected-keyword-arg
    sampler.run_mcmc(pos, draws, store=True)
    return sampler


# pylint:disable=no-member,no-value-for-parameter,invalid-name
def _pyro_noncentered_model(J, sigma, y=None):
    import pyro
    import pyro.distributions as dist

    mu = pyro.sample("mu", dist.Normal(0, 5))
    tau = pyro.sample("tau", dist.HalfCauchy(5))
    with pyro.plate("J", J):
        eta = pyro.sample("eta", dist.Normal(0, 1))
        theta = mu + tau * eta
        return pyro.sample("obs", dist.Normal(theta, sigma), obs=y)


def pyro_noncentered_schools(data, draws, chains):
    """Non-centered eight schools implementation in Pyro."""
    import torch
    from pyro.infer import MCMC, NUTS

    y = torch.from_numpy(data["y"]).float()
    sigma = torch.from_numpy(data["sigma"]).float()

    nuts_kernel = NUTS(_pyro_noncentered_model, jit_compile=True, ignore_jit_warnings=True)
    posterior = MCMC(nuts_kernel, num_samples=draws, warmup_steps=draws, num_chains=chains)
    posterior.run(data["J"], sigma, y)

    # This block lets the posterior be pickled
    posterior.sampler = None
    posterior.kernel.potential_fn = None
    return posterior


# pylint:disable=no-member,no-value-for-parameter,invalid-name
def _numpyro_noncentered_model(J, sigma, y=None):
    import numpyro
    import numpyro.distributions as dist

    mu = numpyro.sample("mu", dist.Normal(0, 5))
    tau = numpyro.sample("tau", dist.HalfCauchy(5))
    with numpyro.plate("J", J):
        eta = numpyro.sample("eta", dist.Normal(0, 1))
        theta = mu + tau * eta
        return numpyro.sample("obs", dist.Normal(theta, sigma), obs=y)


def numpyro_schools_model(data, draws, chains):
    """Centered eight schools implementation in NumPyro."""
    from jax.random import PRNGKey
    from numpyro.infer import MCMC, NUTS

    mcmc = MCMC(
        NUTS(_numpyro_noncentered_model),
        num_warmup=draws,
        num_samples=draws,
        num_chains=chains,
        chain_method="sequential",
    )
    mcmc.run(PRNGKey(0), extra_fields=("num_steps", "energy"), **data)

    # This block lets the posterior be pickled
    mcmc.sampler._sample_fn = None  # pylint: disable=protected-access
    mcmc.sampler._init_fn = None  # pylint: disable=protected-access
    mcmc.sampler._postprocess_fn = None  # pylint: disable=protected-access
    mcmc.sampler._potential_fn = None  # pylint: disable=protected-access
    mcmc.sampler._potential_fn_gen = None  # pylint: disable=protected-access
    mcmc._cache = {}  # pylint: disable=protected-access
    return mcmc


def pystan_noncentered_schools(data, draws, chains):
    """Non-centered eight schools implementation for pystan."""
    schools_code = """
        data {
            int<lower=0> J;
            array[J] real y;
            array[J] real<lower=0> sigma;
        }

        parameters {
            real mu;
            real<lower=0> tau;
            array[J] real eta;
        }

        transformed parameters {
            array[J] real theta;
            for (j in 1:J)
                theta[j] = mu + tau * eta[j];
        }

        model {
            mu ~ normal(0, 5);
            tau ~ cauchy(0, 5);
            eta ~ normal(0, 1);
            y ~ normal(theta, sigma);
        }

        generated quantities {
            array[J] real log_lik;
            array[J] real y_hat;
            for (j in 1:J) {
                log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
                y_hat[j] = normal_rng(theta[j], sigma[j]);
            }
        }
    """
    import stan  # pylint: disable=import-error

    stan_model = stan.build(schools_code, data=data)
    fit = stan_model.sample(num_chains=chains, num_samples=draws, num_warmup=500, save_warmup=True)
    return stan_model, fit


def library_handle(library):
    """Import a library and return the handle."""
    if library == "pystan":
        return importlib.import_module("stan")
    return importlib.import_module(library)


def load_cached_models(eight_schools_data, draws, chains, libs=None):
    """Load pystan, emcee, and pyro models from pickle."""
    here = os.path.dirname(os.path.abspath(__file__))
    supported = (
        # ("pystan", pystan_noncentered_schools),
        ("emcee", emcee_schools_model),
        # ("pyro", pyro_noncentered_schools),
        ("numpyro", numpyro_schools_model),
    )
    data_directory = os.path.join(here, "saved_models")
    if not os.path.isdir(data_directory):
        os.mkdir(data_directory)
    models = {}

    if isinstance(libs, str):
        libs = [libs]

    for library_name, func in supported:
        if libs is not None and library_name not in libs:
            continue
        library = library_handle(library_name)
        if library.__name__ == "stan":
            # PyStan3 does not support pickling
            # httpstan caches models automatically
            _log.info("Generating and loading stan model")
            models["pystan"] = func(eight_schools_data, draws, chains)
            continue

        py_version = sys.version_info
        fname = (
            f"{py_version.major}.{py_version.minor}_{library.__name__}_{library.__version__}"
            f"_{sys.platform}_{draws}_{chains}.pkl.gzip"
        )

        path = os.path.join(data_directory, fname)
        if not os.path.exists(path):
            with gzip.open(path, "wb") as buff:
                try:
                    _log.info("Generating and caching %s", fname)
                    cloudpickle.dump(func(eight_schools_data, draws, chains), buff)
                except AttributeError as err:
                    raise AttributeError(f"Failed caching {library_name}") from err

        with gzip.open(path, "rb") as buff:
            _log.info("Loading %s from cache", fname)
            models[library.__name__] = cloudpickle.load(buff)

    return models


def test_precompile_models(eight_schools_params, draws, chains):
    """Precompile model files."""
    load_cached_models(eight_schools_params, draws, chains)


def running_on_ci() -> bool:
    """Return True if running on CI machine."""
    return os.environ.get("ARVIZ_CI_MACHINE") is not None


def importorskip(modname: str, minversion: str | None = None, reason: str | None = None) -> Any:
    """Import and return the requested module ``modname``.

        Doesn't allow skips on CI machine.
        Borrowed and modified from ``pytest.importorskip``.
    :param str modname: the name of the module to import
    :param str minversion: if given, the imported module's ``__version__``
        attribute must be at least this minimal version, otherwise the test is
        still skipped.
    :param str reason: if given, this reason is shown as the message when the
        module cannot be imported.
    :returns: The imported module. This should be assigned to its canonical
        name.
    Example::
        docutils = pytest.importorskip("docutils")
    """
    # ARVIZ_CI_MACHINE is True if tests run on CI, where ARVIZ_CI_MACHINE env variable exists
    ARVIZ_CI_MACHINE = running_on_ci()
    if not ARVIZ_CI_MACHINE:
        return pytest.importorskip(modname=modname, minversion=minversion, reason=reason)
    import warnings

    compile(modname, "", "eval")  # to catch syntaxerrors

    with warnings.catch_warnings():
        # make sure to ignore ImportWarnings that might happen because
        # of existing directories with the same name we're trying to
        # import but without a __init__.py file
        warnings.simplefilter("ignore")
        __import__(modname)
    mod = sys.modules[modname]
    if minversion is None:
        return mod
    verattr = getattr(mod, "__version__", None)
    if verattr is None or Version(verattr) < Version(minversion):
        raise Skipped(
            f"module {modname} has __version__ {verattr}, required is: {minversion}",
            allow_module_level=True,
        )
    return mod


@pytest.fixture(scope="module")
def eight_schools_params():
    """Share setup for eight schools."""
    return {
        "J": 8,
        "y": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
        "sigma": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
    }


@pytest.fixture(scope="module")
def draws():
    """Share default draw count."""
    return 500


@pytest.fixture(scope="module")
def chains():
    """Share default chain count."""
    return 2
