"""ArviZ testing utilities."""

import numpy as np

from arviz_base import from_dict


def generate_base_data(seed=31):
    """Generate a base dataset for testing."""
    from scipy.stats import halfnorm, norm

    rng = np.random.default_rng(seed)
    mu = rng.normal(loc=1, size=(4, 100))
    tau = np.exp(rng.normal(size=(4, 100)))
    theta_orig = rng.uniform(size=7)
    theta = rng.normal(theta_orig[None, None, :], scale=1, size=(4, 100, 7))
    idxs = rng.choice(np.arange(7), size=29)
    x = np.linspace(0, 1, 29)
    obs = rng.normal(loc=x + theta_orig[idxs], scale=3)
    log_lik = norm(mu[:, :, None] * x[None, None, :] + theta[:, :, idxs], tau[:, :, None]).logpdf(
        obs[None, None, :]
    )
    log_lik = log_lik / log_lik.var()
    log_mu_prior = norm(0, 3).logpdf(mu)
    log_tau_prior = halfnorm(scale=5).logpdf(tau)
    log_theta_prior = norm(0, 1).logpdf(theta)
    prior_predictive = rng.normal(size=(1, 100, 29))
    posterior_predictive = rng.normal(size=(4, 100, 29))
    diverging = rng.choice([True, False], size=(4, 100), p=[0.1, 0.9])
    mu_prior = norm(0, 3).rvs(size=(1, 500), random_state=rng)
    tau_prior = halfnorm(0, 5).rvs(size=(1, 500), random_state=rng)
    theta_prior = norm(0, 1).rvs(size=(1, 500, 7), random_state=rng)
    energy = rng.normal(loc=50, scale=10, size=(4, 100))

    return {
        "posterior": {"mu": mu, "theta": theta, "tau": tau},
        "observed_data": {"y": obs},
        "log_likelihood": {"y": log_lik},
        "log_prior": {"mu": log_mu_prior, "theta": log_theta_prior, "tau": log_tau_prior},
        "prior_predictive": {"y": prior_predictive},
        "posterior_predictive": {"y": posterior_predictive},
        "sample_stats": {"diverging": diverging, "energy": energy},
        "prior": {"mu": mu_prior, "theta": theta_prior, "tau": tau_prior},
    }


def datatree(seed=31):
    """Generate a general DataTree."""
    base_data = generate_base_data(seed)

    dt = from_dict(
        base_data,
        dims={"theta": ["hierarchy"], "y": ["obs_dim"]},
    )
    dt["point_estimate"] = dt.posterior.mean(("chain", "draw"))
    dt["trunk"] = dt.azstats.eti(prob=0.5)
    dt["twig"] = dt.azstats.eti(prob=0.9)
    return dt


def datatree2(seed=17):
    """Generate a DataTree with a posterior and sample stats."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(4, 100))
    tau = rng.normal(size=(4, 100))
    theta = rng.normal(size=(4, 100, 7))
    theta_t = rng.normal(size=(4, 100, 7))
    diverging = rng.choice([True, False], size=(4, 100), p=[0.1, 0.9])

    return from_dict(
        {
            "posterior": {"mu": mu, "theta": theta, "tau": tau, "theta_t": theta_t},
            "sample_stats": {"diverging": diverging},
        },
        dims={"theta": ["hierarchy"], "theta_t": ["hierarchy"]},
    )


def datatree3(seed=17):
    """Generate a DataTree with discrete data."""
    rng = np.random.default_rng(seed)
    posterior_predictive = rng.poisson(4, size=(4, 100, 7))
    observed_data = rng.poisson(4, size=7)

    return from_dict(
        {
            "posterior_predictive": {"y": posterior_predictive},
            "observed_data": {"y": observed_data},
        },
        dims={"y": ["obs_dim"]},
    )


def datatree_binary(seed=17):
    """Generate a DataTree with binary data."""
    rng = np.random.default_rng(seed)
    posterior_predictive = rng.binomial(1, 0.5, size=(4, 100, 7))
    observed_data = rng.binomial(1, 0.5, size=7)
    log_likelihood = rng.normal(loc=0, scale=1, size=(4, 100, 7))

    return from_dict(
        {
            "posterior_predictive": {"y": posterior_predictive},
            "observed_data": {"y": observed_data},
            "log_likelihood": {"y": log_likelihood},
        },
        dims={"y": ["obs_dim"]},
    )


def datatree_regression(seed=17):
    """Generate a DataTree for regression data."""
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    n_obs = 100
    true_sigma = 0.9
    true_mu = 2 * np.linspace(-1, 1, n_obs)
    observed_data = true_mu + rng.normal(0, true_sigma, size=n_obs)

    posterior_sigma = rng.normal(true_sigma, 0.1, size=(4, 500))
    posterior_sigma = np.abs(posterior_sigma)

    posterior_mu = rng.normal(true_mu, true_sigma * 0.5, size=(4, 500, n_obs))
    posterior_predictive = rng.normal(posterior_mu, true_sigma, size=(4, 500, n_obs))
    log_likelihood = norm(posterior_mu, true_sigma).logpdf(observed_data)

    return from_dict(
        {
            "posterior": {"mu": posterior_mu, "sigma": posterior_sigma},
            "posterior_predictive": {"y": posterior_predictive},
            "observed_data": {"y": observed_data},
            "log_likelihood": {"y": log_likelihood},
        },
        dims={"y": ["obs_dim"]},
    )


def datatree_4d(seed=31):
    """Generate a DataTree with a 4D posterior."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(4, 100))
    theta = rng.normal(size=(4, 100, 5))
    eta = rng.normal(size=(4, 100, 5, 3))
    diverging = rng.choice([True, False], size=(4, 100), p=[0.1, 0.9])
    obs = rng.normal(size=(5, 3))
    prior_predictive = rng.normal(size=(1, 100, 5, 3))
    posterior_predictive = rng.normal(size=(4, 100, 5, 3))

    return from_dict(
        {
            "posterior": {"mu": mu, "theta": theta, "eta": eta},
            "observed_data": {"obs": obs},
            "prior_predictive": {"obs": prior_predictive},
            "posterior_predictive": {"obs": posterior_predictive},
            "sample_stats": {"diverging": diverging},
        },
        dims={"theta": ["hierarchy"], "eta": ["hierarchy", "group"]},
    )


def datatree_sample(seed=31):
    """Generate a DataTree with sample dimensions."""
    base_data = generate_base_data(seed)

    return from_dict(
        {
            group: {
                key: values[0] if group != "observed_data" else values
                for key, values in group_dict.items()
            }
            for group, group_dict in base_data.items()
        },
        dims={"theta": ["hierarchy"]},
        sample_dims=["sample"],
    )


def cmp():
    """Generate a comparison DataFrame."""
    import pandas as pd

    return pd.DataFrame(
        {
            "elpd": [-4.5, -14.3, -16.2],
            "p": [2.6, 2.3, 2.1],
            "elpd_diff": [0, 9.7, 11.3],
            "weight": [0.9, 0.1, 0],
            "se": [2.3, 2.7, 2.3],
            "dse": [0, 2.7, 2.3],
            "warning": [False, False, False],
        },
        index=["Model B", "Model A", "Model C"],
    )


def fake_dt():
    """Generate a fake prior/posterior DataTree."""
    rng = np.random.default_rng(42)

    return from_dict(
        {
            "posterior": {
                "a": rng.normal(size=(4, 100)),
                "b": rng.normal(size=(4, 100)),
            },
            "prior": {
                "a": rng.normal(size=(4, 100)),
                "b": rng.normal(size=(4, 100)),
            },
        }
    )


def datatree_censored(seed=42):
    """Create a sample DataTree with censored data."""
    rng = np.random.default_rng(seed)

    # Parameters
    n_obs = 100
    n_samples = 1000
    n_chains = 4
    observed_times = rng.exponential(scale=8, size=n_obs)
    status = rng.binomial(1, 0.7, size=n_obs)

    posterior_predictive_times = rng.exponential(scale=8, size=(n_chains, n_samples, n_obs))

    # Create DataTree
    return from_dict(
        {
            "observed_data": {
                "time": observed_times,
            },
            "constant_data": {
                "time": status,
            },
            "posterior_predictive": {
                "time": posterior_predictive_times,
            },
        },
        dims={"time": ["subject"]},
    )


def check_multiple_attrs(test_dict, parent):
    """Perform multiple hasattr checks on InferenceData objects.

    It is thought to first check if the parent object contains a given dataset,
    and then (if present) check the attributes of the dataset.

    Given the output of the function, all mismatches between expectation and reality can
    be retrieved: a single string indicates a group mismatch and a tuple of strings
    ``(group, var)`` indicates a mismatch in the variable ``var`` of ``group``.

    Parameters
    ----------
    test_dict : dict of {str : list of str}
        Its structure should be `{dataset1_name: [var1, var2], dataset2_name: [var]}`.
        A ``~`` at the beginning of a dataset or variable name indicates the name NOT
        being present must be asserted.
    parent : DataTree
        DataTree object on which to check the attributes.

    Returns
    -------
    list of (str or tuple of (str, str))
        List containing the failed checks. It will contain either the dataset_name or a
        tuple (dataset_name, var) for all non present attributes.

    Examples
    --------
    The output below indicates that ``posterior`` group was expected but not found, and
    variables ``a`` and ``b``:

        ["posterior", ("prior", "a"), ("prior", "b")]

    Another example could be the following:

        [("posterior", "a"), "~observed_data", ("sample_stats", "~log_likelihood")]

    In this case, the output indicates that variable ``a`` was not found in ``posterior``
    as it was expected, however, in the other two cases, the preceding ``~`` (kept from the
    input negation notation) indicates that ``observed_data`` group should not be present
    but was found in the InferenceData and that ``log_likelihood`` variable was found
    in ``sample_stats``, also against what was expected.

    """
    failed_attrs: list[str | tuple[str, str]] = []
    for dataset_name, attributes in test_dict.items():
        if dataset_name.startswith("~"):
            if hasattr(parent, dataset_name[1:]):
                failed_attrs.append(dataset_name)
        elif hasattr(parent, dataset_name):
            dataset = getattr(parent, dataset_name)
            for attribute in attributes:
                if attribute.startswith("~"):
                    if hasattr(dataset, attribute[1:]):
                        failed_attrs.append((dataset_name, attribute))
                elif not hasattr(dataset, attribute):
                    failed_attrs.append((dataset_name, attribute))
        else:
            failed_attrs.append(dataset_name)
    return failed_attrs
