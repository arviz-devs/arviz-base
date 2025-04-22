#  pylint: disable=too-many-instance-attributes,too-many-lines
"""PyStan-specific conversion code."""

import logging
import re
from collections import OrderedDict
from copy import deepcopy
from math import ceil

import numpy as np
from xarray import DataArray, Dataset, DataTree

from arviz_base.base import (
    dict_to_dataset,
    generate_dims_coords,
    infer_stan_dtypes,
    make_attrs,
    requires,
)
from arviz_base.rcparams import rcParams

_log = logging.getLogger(__name__)

try:
    import ujson as json
except ImportError:
    # Can't find ujson using json
    # mypy struggles with conditional imports expressed as catching ImportError:
    # https://github.com/python/mypy/issues/1153
    import json  # type: ignore


class PyStan3Converter:
    """Encapsulate PyStan3 specific logic."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        *,
        posterior=None,
        posterior_model=None,
        posterior_predictive=None,
        predictions=None,
        prior=None,
        prior_model=None,
        prior_predictive=None,
        observed_data=None,
        constant_data=None,
        predictions_constant_data=None,
        log_likelihood=None,
        coords=None,
        dims=None,
        save_warmup=None,
        dtypes=None,
    ):
        self.posterior = posterior
        self.posterior_model = posterior_model
        self.posterior_predictive = posterior_predictive
        self.predictions = predictions
        self.prior = prior
        self.prior_model = prior_model
        self.prior_predictive = prior_predictive
        self.observed_data = observed_data
        self.constant_data = constant_data
        self.predictions_constant_data = predictions_constant_data
        self.log_likelihood = (
            rcParams["data.log_likelihood"] if log_likelihood is None else log_likelihood
        )
        self.coords = coords
        self.dims = dims
        self.save_warmup = rcParams["data.save_warmup"] if save_warmup is None else save_warmup
        self.dtypes = dtypes

        if (
            self.log_likelihood is True
            and self.posterior is not None
            and "log_lik" in self.posterior.param_names
        ):
            self.log_likelihood = ["log_lik"]
        elif isinstance(self.log_likelihood, bool):
            self.log_likelihood = None

        import stan  # pylint: disable=import-error

        self.stan = stan

    def _warmup_return_to_dict(self, data, data_warmup, group, attrs=None):
        res = {
            group: dict_to_dataset(
                data,
                inference_library=self.stan,
                coords=self.coords,
                dims=self.dims,
                skip_event_dims="log_likelihood" == group,
                attrs=attrs,
            ),
        }
        if self.save_warmup and data_warmup:
            res[f"warmup_{group}"] = dict_to_dataset(
                data_warmup,
                inference_library=self.stan,
                coords=self.coords,
                dims=self.dims,
                skip_event_dims="log_likelihood" == group,
                attrs=attrs,
            )
        return res

    @requires("posterior")
    def posterior_to_xarray(self):
        """Extract posterior samples from fit."""
        posterior = self.posterior
        posterior_model = self.posterior_model
        # filter posterior_predictive and log_likelihood
        posterior_predictive = self.posterior_predictive
        if posterior_predictive is None:
            posterior_predictive = []
        elif isinstance(posterior_predictive, str):
            posterior_predictive = [posterior_predictive]
        predictions = self.predictions
        if predictions is None:
            predictions = []
        elif isinstance(predictions, str):
            predictions = [predictions]
        log_likelihood = self.log_likelihood
        if log_likelihood is None:
            log_likelihood = []
        elif isinstance(log_likelihood, str):
            log_likelihood = [log_likelihood]
        elif isinstance(log_likelihood, dict):
            log_likelihood = list(log_likelihood.values())

        ignore = posterior_predictive + predictions + log_likelihood

        data, data_warmup = get_draws_stan3(
            posterior,
            model=posterior_model,
            ignore=ignore,
            warmup=self.save_warmup,
            dtypes=self.dtypes,
        )
        attrs = get_attrs_stan3(posterior, model=posterior_model)
        return self._warmup_return_to_dict(data, data_warmup, "posterior", attrs)

    @requires("posterior")
    def sample_stats_to_xarray(self):
        """Extract sample_stats from posterior."""
        posterior = self.posterior
        posterior_model = self.posterior_model
        data, data_warmup = get_sample_stats_stan3(
            posterior, ignore="lp__", warmup=self.save_warmup, dtypes=self.dtypes
        )
        data_lp, data_warmup_lp = get_sample_stats_stan3(
            posterior, variables="lp__", warmup=self.save_warmup
        )
        data["lp"] = data_lp["lp"]
        if data_warmup_lp:
            data_warmup["lp"] = data_warmup_lp["lp"]

        attrs = get_attrs_stan3(posterior, model=posterior_model)
        return self._warmup_return_to_dict(data, data_warmup, "sample_stats", attrs)

    @requires("posterior")
    @requires("log_likelihood")
    def log_likelihood_to_xarray(self):
        """Store log_likelihood data in log_likelihood group."""
        fit = self.posterior

        log_likelihood = self.log_likelihood
        model = self.posterior_model
        if isinstance(log_likelihood, str):
            log_likelihood = [log_likelihood]
        if isinstance(log_likelihood, (list | tuple)):
            log_likelihood = {name: name for name in log_likelihood}
        log_likelihood_draws, log_likelihood_draws_warmup = get_draws_stan3(
            fit,
            model=model,
            variables=list(log_likelihood.values()),
            warmup=self.save_warmup,
            dtypes=self.dtypes,
        )
        data = {
            obs_var_name: log_likelihood_draws[log_like_name]
            for obs_var_name, log_like_name in log_likelihood.items()
            if log_like_name in log_likelihood_draws
        }
        data_warmup = {
            obs_var_name: log_likelihood_draws_warmup[log_like_name]
            for obs_var_name, log_like_name in log_likelihood.items()
            if log_like_name in log_likelihood_draws_warmup
        }
        return self._warmup_return_to_dict(data, data_warmup, "log_likelihood")

    @requires("posterior")
    @requires("posterior_predictive")
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        posterior = self.posterior
        posterior_model = self.posterior_model
        posterior_predictive = self.posterior_predictive
        data, data_warmup = get_draws_stan3(
            posterior,
            model=posterior_model,
            variables=posterior_predictive,
            warmup=self.save_warmup,
            dtypes=self.dtypes,
        )
        return self._warmup_return_to_dict(data, data_warmup, "posterior_predictive")

    @requires("posterior")
    @requires("predictions")
    def predictions_to_xarray(self):
        """Convert predictions samples to xarray."""
        posterior = self.posterior
        posterior_model = self.posterior_model
        predictions = self.predictions
        data, data_warmup = get_draws_stan3(
            posterior,
            model=posterior_model,
            variables=predictions,
            warmup=self.save_warmup,
            dtypes=self.dtypes,
        )
        return self._warmup_return_to_dict(data, data_warmup, "predictions")

    @requires("prior")
    def prior_to_xarray(self):
        """Convert prior samples to xarray."""
        prior = self.prior
        prior_model = self.prior_model
        # filter posterior_predictive and log_likelihood
        prior_predictive = self.prior_predictive
        if prior_predictive is None:
            prior_predictive = []
        elif isinstance(prior_predictive, str):
            prior_predictive = [prior_predictive]

        ignore = prior_predictive

        data, data_warmup = get_draws_stan3(
            prior, model=prior_model, ignore=ignore, warmup=self.save_warmup, dtypes=self.dtypes
        )
        attrs = get_attrs_stan3(prior, model=prior_model)
        return self._warmup_return_to_dict(data, data_warmup, "prior", attrs)

    @requires("prior")
    def sample_stats_prior_to_xarray(self):
        """Extract sample_stats_prior from prior."""
        prior = self.prior
        prior_model = self.prior_model
        data, data_warmup = get_sample_stats_stan3(
            prior, warmup=self.save_warmup, dtypes=self.dtypes
        )
        attrs = get_attrs_stan3(prior, model=prior_model)
        return self._warmup_return_to_dict(data, data_warmup, "sample_stats_prior", attrs)

    @requires("prior")
    @requires("prior_predictive")
    def prior_predictive_to_xarray(self):
        """Convert prior_predictive samples to xarray."""
        prior = self.prior
        prior_model = self.prior_model
        prior_predictive = self.prior_predictive
        data, data_warmup = get_draws_stan3(
            prior,
            model=prior_model,
            variables=prior_predictive,
            warmup=self.save_warmup,
            dtypes=self.dtypes,
        )
        return self._warmup_return_to_dict(data, data_warmup, "prior_predictive")

    @requires("posterior_model")
    @requires(["observed_data"])
    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        posterior_model = self.posterior_model
        dims = {} if self.dims is None else self.dims
        names = getattr(self, "observed_data")
        if names is None:
            return None
        names = [names] if isinstance(names, str) else names
        data = OrderedDict()
        for key in names:
            vals = np.atleast_1d(posterior_model.data[key])
            val_dims = dims.get(key)
            val_dims, coords = generate_dims_coords(
                vals.shape, key, dims=val_dims, coords=self.coords
            )
            data[key] = DataArray(vals, dims=val_dims, coords=coords)
        return Dataset(data_vars=data, attrs=make_attrs(inference_library=self.stan))

    @requires("posterior_model")
    @requires(["constant_data"])
    def constant_data_to_xarray(self):
        """Convert observed data to xarray."""
        posterior_model = self.posterior_model
        dims = {} if self.dims is None else self.dims
        names = getattr(self, "constant_data")
        if names is None:
            return None
        names = [names] if isinstance(names, str) else names
        data = OrderedDict()
        for key in names:
            vals = np.atleast_1d(posterior_model.data[key])
            val_dims = dims.get(key)
            val_dims, coords = generate_dims_coords(
                vals.shape, key, dims=val_dims, coords=self.coords
            )
            data[key] = DataArray(vals, dims=val_dims, coords=coords)
        return Dataset(data_vars=data, attrs=make_attrs(inference_library=self.stan))

    @requires("posterior_model")
    @requires("predictions_constant_data")
    def predictions_constant_data_to_xarray(self):
        """Convert observed data to xarray."""
        posterior_model = self.posterior_model
        dims = {} if self.dims is None else self.dims
        names = self.predictions_constant_data
        names = [names] if isinstance(names, str) else names
        data = OrderedDict()
        for key in names:
            vals = np.atleast_1d(posterior_model.data[key])
            val_dims = dims.get(key)
            val_dims, coords = generate_dims_coords(
                vals.shape, key, dims=val_dims, coords=self.coords
            )
            data[key] = DataArray(vals, dims=val_dims, coords=coords)
        return Dataset(data_vars=data, attrs=make_attrs(inference_library=self.stan))

    def to_datatree(self):
        """Convert all available data to a DataTree object.

        Note that if groups can not be created (i.e., there is no `output`, so
        the `posterior` and `sample_stats` can not be extracted), then the DataTree
        will not have those groups.
        """
        datadict = {
            "observed_data": self.observed_data_to_xarray(),
            "constant_data": self.constant_data_to_xarray(),
            "predictions_constant_data": self.predictions_constant_data_to_xarray(),
        }
        datalist = [
            self.posterior_to_xarray(),
            self.sample_stats_to_xarray(),
            self.posterior_predictive_to_xarray(),
            self.predictions_to_xarray(),
            self.prior_to_xarray(),
            self.sample_stats_prior_to_xarray(),
            self.prior_predictive_to_xarray(),
            self.log_likelihood_to_xarray(),
        ]
        for ds_dict in datalist:
            if ds_dict is not None:
                datadict.update(ds_dict)
        return DataTree.from_dict({group: ds for group, ds in datadict.items() if ds is not None})


def get_sample_stats(fit, warmup=False, dtypes=None):
    """Extract sample stats from PyStan fit."""
    if dtypes is None:
        dtypes = {}
    dtypes = {"divergent__": bool, "n_leapfrog__": np.int64, "treedepth__": np.int64, **dtypes}

    rename_dict = {
        "divergent": "diverging",
        "n_leapfrog": "n_steps",
        "treedepth": "tree_depth",
        "stepsize": "step_size",
        "accept_stat": "acceptance_rate",
    }

    ndraws_warmup = fit.sim["warmup2"]
    if max(ndraws_warmup) == 0:
        warmup = False
    ndraws = [s - w for s, w in zip(fit.sim["n_save"], ndraws_warmup)]

    extraction = OrderedDict()
    extraction_warmup = OrderedDict()
    for chain, (pyholder, ndraw, ndraw_warmup) in enumerate(
        zip(fit.sim["samples"], ndraws, ndraws_warmup)
    ):
        if chain == 0:
            for key in pyholder["sampler_param_names"]:
                extraction[key] = []
                if warmup:
                    extraction_warmup[key] = []
        for key, values in zip(pyholder["sampler_param_names"], pyholder["sampler_params"]):
            extraction[key].append(values[-ndraw:])
            if warmup:
                extraction_warmup[key].append(values[:ndraw_warmup])

    data = OrderedDict()
    for key, values in extraction.items():
        vals = np.stack(values, axis=0)
        dtype = dtypes.get(key)
        vals = vals.astype(dtype)
        name = re.sub("__$", "", key)
        name = rename_dict.get(name, name)
        data[name] = vals

    data_warmup = OrderedDict()
    if warmup:
        for key, values in extraction_warmup.items():
            vals = np.stack(values, axis=0)
            vals = vals.astype(dtypes.get(key))
            name = re.sub("__$", "", key)
            name = rename_dict.get(name, name)
            data_warmup[name] = vals

    return data, data_warmup


def get_attrs(fit):
    """Get attributes from PyStan fit object."""
    attrs = {}

    try:
        attrs["args"] = [deepcopy(holder.args) for holder in fit.sim["samples"]]
    except Exception as exp:  # pylint: disable=broad-except
        _log.warning("Failed to fetch args from fit: %s", exp)
    if "args" in attrs:
        for arg in attrs["args"]:
            if isinstance(arg["init"], bytes):
                arg["init"] = arg["init"].decode("utf-8")
        attrs["args"] = json.dumps(attrs["args"])
    try:
        attrs["inits"] = [holder.inits for holder in fit.sim["samples"]]
    except Exception as exp:  # pylint: disable=broad-except
        _log.warning("Failed to fetch `args` from fit: %s", exp)
    else:
        attrs["inits"] = json.dumps(attrs["inits"])

    attrs["step_size"] = []
    attrs["metric"] = []
    attrs["inv_metric"] = []
    for holder in fit.sim["samples"]:
        try:
            step_size = float(
                re.search(
                    r"step\s*size\s*=\s*([0-9]+.?[0-9]+)\s*",
                    holder.adaptation_info,
                    flags=re.IGNORECASE,
                ).group(1)
            )
        except AttributeError:
            step_size = np.nan
        attrs["step_size"].append(step_size)

        inv_metric_match = re.search(
            r"mass matrix:\s*(.*)\s*$", holder.adaptation_info, flags=re.DOTALL
        )
        if inv_metric_match:
            inv_metric_str = inv_metric_match.group(1)
            if "Diagonal elements of inverse mass matrix" in holder.adaptation_info:
                metric = "diag_e"
                inv_metric = [float(item) for item in inv_metric_str.strip(" #\n").split(",")]
            else:
                metric = "dense_e"
                inv_metric = [
                    list(map(float, item.split(",")))
                    for item in re.sub(r"#\s", "", inv_metric_str).splitlines()
                ]
        else:
            metric = "unit_e"
            inv_metric = None

        attrs["metric"].append(metric)
        attrs["inv_metric"].append(inv_metric)
    attrs["inv_metric"] = json.dumps(attrs["inv_metric"])

    if not attrs["step_size"]:
        del attrs["step_size"]

    attrs["adaptation_info"] = fit.get_adaptation_info()
    attrs["stan_code"] = fit.get_stancode()

    return attrs


def get_draws_stan3(fit, model=None, variables=None, ignore=None, warmup=False, dtypes=None):
    """Extract draws from PyStan3 fit."""
    if ignore is None:
        ignore = []

    if dtypes is None:
        dtypes = {}

    if model is not None:
        dtypes = {**infer_dtypes(fit, model), **dtypes}

    if not fit.save_warmup:
        warmup = False

    num_warmup = ceil((fit.num_warmup * fit.save_warmup) / fit.num_thin)

    if variables is None:
        variables = fit.param_names
    elif isinstance(variables, str):
        variables = [variables]
    variables = list(variables)

    data = OrderedDict()
    data_warmup = OrderedDict()

    for var in variables:
        if var in ignore:
            continue
        if var in data:
            continue
        dtype = dtypes.get(var)

        new_shape = (*fit.dims[fit.param_names.index(var)], -1, fit.num_chains)
        if 0 in new_shape:
            continue
        values = fit._draws[fit._parameter_indexes(var), :]  # pylint: disable=protected-access
        values = values.reshape(new_shape, order="F")
        values = np.moveaxis(values, [-2, -1], [1, 0])
        values = values.astype(dtype)
        if warmup:
            data_warmup[var] = values[:, num_warmup:]
        data[var] = values[:, num_warmup:]

    return data, data_warmup


def get_sample_stats_stan3(fit, variables=None, ignore=None, warmup=False, dtypes=None):
    """Extract sample stats from PyStan3 fit."""
    if dtypes is None:
        dtypes = {}
    dtypes = {"divergent__": bool, "n_leapfrog__": np.int64, "treedepth__": np.int64, **dtypes}

    rename_dict = {
        "divergent": "diverging",
        "n_leapfrog": "n_steps",
        "treedepth": "tree_depth",
        "stepsize": "step_size",
        "accept_stat": "acceptance_rate",
    }

    if isinstance(variables, str):
        variables = [variables]
    if isinstance(ignore, str):
        ignore = [ignore]

    if not fit.save_warmup:
        warmup = False

    num_warmup = ceil((fit.num_warmup * fit.save_warmup) / fit.num_thin)

    data = OrderedDict()
    data_warmup = OrderedDict()
    for key in fit.sample_and_sampler_param_names:
        if (variables and key not in variables) or (ignore and key in ignore):
            continue
        new_shape = -1, fit.num_chains
        values = fit._draws[fit._parameter_indexes(key)]  # pylint: disable=protected-access
        values = values.reshape(new_shape, order="F")
        values = np.moveaxis(values, [-2, -1], [1, 0])
        dtype = dtypes.get(key)
        values = values.astype(dtype)
        name = re.sub("__$", "", key)
        name = rename_dict.get(name, name)
        if warmup:
            data_warmup[name] = values[:, :num_warmup]
        data[name] = values[:, num_warmup:]

    return data, data_warmup


def get_attrs_stan3(fit, model=None):
    """Get attributes from PyStan3 fit and model object."""
    attrs = {}
    for key in ["num_chains", "num_samples", "num_thin", "num_warmup", "save_warmup"]:
        try:
            attrs[key] = getattr(fit, key)
        except AttributeError as exp:
            _log.warning("Failed to access attribute %s in fit object %s", key, exp)

    if model is not None:
        for key in ["model_name", "program_code", "random_seed"]:
            try:
                attrs[key] = getattr(model, key)
            except AttributeError as exp:
                _log.warning("Failed to access attribute %s in model object %s", key, exp)

    return attrs


def infer_dtypes(fit, model=None):
    """Infer dtypes from Stan model code.

    Function strips out generated quantities block and searches for `int`
    dtypes after stripping out comments inside the block.
    """
    if model is None:
        stan_code = fit.get_stancode()
        model_pars = fit.model_pars
    else:
        stan_code = model.program_code
        model_pars = fit.param_names

    dtypes = {key: item for key, item in infer_stan_dtypes(stan_code).items() if key in model_pars}
    return dtypes


# pylint disable=too-many-instance-attributes
def from_pystan(
    posterior=None,
    *,
    posterior_predictive=None,
    predictions=None,
    prior=None,
    prior_predictive=None,
    observed_data=None,
    constant_data=None,
    predictions_constant_data=None,
    log_likelihood=None,
    coords=None,
    dims=None,
    posterior_model=None,
    prior_model=None,
    save_warmup=None,
    dtypes=None,
):
    """Convert PyStan data into an InferenceData object.

    For a usage example read the
    :ref:`Creating InferenceData section on from_pystan <creating_InferenceData>`

    Parameters
    ----------
    posterior : StanFit4Model or stan.fit.Fit
        PyStan fit object for posterior.
    posterior_predictive : str, a list of str
        Posterior predictive samples for the posterior.
    predictions : str, a list of str
        Out-of-sample predictions for the posterior.
    prior : StanFit4Model or stan.fit.Fit
        PyStan fit object for prior.
    prior_predictive : str, a list of str
        Posterior predictive samples for the prior.
    observed_data : str or a list of str
        observed data used in the sampling.
        Observed data is extracted from the `posterior.data`.
        PyStan3 needs model object for the extraction.
        See `posterior_model`.
    constant_data : str or list of str
        Constants relevant to the model (i.e. x values in a linear
        regression).
    predictions_constant_data : str or list of str
        Constants relevant to the model predictions (i.e. new x values in a linear
        regression).
    log_likelihood : dict of {str: str}, list of str or str, optional
        Pointwise log_likelihood for the data. log_likelihood is extracted from the
        posterior. It is recommended to use this argument as a dictionary whose keys
        are observed variable names and its values are the variables storing log
        likelihood arrays in the Stan code. In other cases, a dictionary with keys
        equal to its values is used. By default, if a variable ``log_lik`` is
        present in the Stan model, it will be retrieved as pointwise log
        likelihood values. Use ``False`` or set ``data.log_likelihood`` to
        false to avoid this behaviour.
    coords : dict[str, iterable]
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict[str, List(str)]
        A mapping from variables to a list of coordinate names for the variable.
    posterior_model : stan.model.Model
        PyStan3 specific model object. Needed for automatic dtype parsing
        and for the extraction of observed data.
    prior_model : stan.model.Model
        PyStan3 specific model object. Needed for automatic dtype parsing.
    save_warmup : bool
        Save warmup iterations into InferenceData object. If not defined, use default
        defined by the rcParams.
    dtypes: dict
        A dictionary containing dtype information (int, float) for parameters.
        By default dtype information is extracted from the model code.
        Model code is extracted from fit object in PyStan 2 and from model object
        in PyStan 3.

    Returns
    -------
    InferenceData object
    """
    return PyStan3Converter(
        posterior=posterior,
        posterior_model=posterior_model,
        posterior_predictive=posterior_predictive,
        predictions=predictions,
        prior=prior,
        prior_model=prior_model,
        prior_predictive=prior_predictive,
        observed_data=observed_data,
        constant_data=constant_data,
        predictions_constant_data=predictions_constant_data,
        log_likelihood=log_likelihood,
        coords=coords,
        dims=dims,
        save_warmup=save_warmup,
        dtypes=dtypes,
    ).to_datatree()
