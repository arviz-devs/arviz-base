"""CmdStanPy specific conversion code."""

import logging
import re
from pathlib import Path

import numpy as np
from datatree import DataTree

from .base import dict_to_dataset, infer_stan_dtypes, requires
from .rcparams import rcParams

_log = logging.getLogger(__name__)


class CmdStanPyConverter:
    """Encapsulate CmdStanPy specific logic."""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        *,
        posterior=None,
        posterior_predictive=None,
        predictions=None,
        prior=None,
        prior_predictive=None,
        observed_data=None,
        constant_data=None,
        predictions_constant_data=None,
        log_likelihood=None,
        index_origin=None,
        coords=None,
        dims=None,
        save_warmup=None,
        dtypes=None,
    ):
        self.posterior = posterior  # CmdStanPy CmdStanMCMC object
        self.posterior_predictive = posterior_predictive
        self.predictions = predictions
        self.prior = prior
        self.prior_predictive = prior_predictive
        self.observed_data = observed_data
        self.constant_data = constant_data
        self.predictions_constant_data = predictions_constant_data
        self.log_likelihood = (
            rcParams["data.log_likelihood"] if log_likelihood is None else log_likelihood
        )
        self.index_origin = index_origin
        self.coords = coords
        self.dims = dims

        self.save_warmup = rcParams["data.save_warmup"] if save_warmup is None else save_warmup

        import cmdstanpy  # pylint: disable=import-error

        if dtypes is None:
            dtypes = {}
        elif isinstance(dtypes, cmdstanpy.model.CmdStanModel):
            model_code = dtypes.code()
            dtypes = infer_stan_dtypes(model_code)
        elif isinstance(dtypes, str):
            dtypes_path = Path(dtypes)
            if dtypes_path.exists():
                with dtypes_path.open("r", encoding="UTF-8") as f_obj:
                    model_code = f_obj.read()
            else:
                model_code = dtypes
            dtypes = infer_stan_dtypes(model_code)

        self.dtypes = dtypes

        if self.log_likelihood is True and "log_lik" in self.posterior.stan_variables():
            self.log_likelihood = ["log_lik"]

        if isinstance(self.log_likelihood, bool):
            self.log_likelihood = None

        self.cmdstanpy = cmdstanpy

    def _warmup_return_to_dict(self, data, data_warmup, group):
        res = {
            group: dict_to_dataset(
                data,
                inference_library=self.cmdstanpy,
                coords=self.coords,
                dims=self.dims,
                index_origin=self.index_origin,
                skip_event_dims="log_likelihood" == group,
            ),
        }
        if self.save_warmup and data_warmup:
            res[f"warmup_{group}"] = dict_to_dataset(
                data_warmup,
                inference_library=self.cmdstanpy,
                coords=self.coords,
                dims=self.dims,
                index_origin=self.index_origin,
                skip_event_dims="log_likelihood" == group,
            )
        return res

    @requires("posterior")
    def posterior_to_xarray(self):
        """Extract posterior samples from output csv."""
        items = list(self.posterior.stan_variables().keys())
        if self.posterior_predictive is not None:
            try:
                items = _filter(items, self.posterior_predictive)
            except ValueError:
                pass
        if self.predictions is not None:
            try:
                items = _filter(items, self.predictions)
            except ValueError:
                pass
        if self.log_likelihood is not None:
            try:
                items = _filter(items, self.log_likelihood)
            except ValueError:
                pass

        data, data_warmup = _unpack_fit(
            self.posterior,
            items,
            self.save_warmup,
            self.dtypes,
        )

        return self._warmup_return_to_dict(data, data_warmup, "posterior")

    @requires("posterior")
    def sample_stats_to_xarray(self):
        """Extract sample_stats from prosterior fit."""
        data, data_warmup = self.stats_to_xarray(self.posterior)
        return self._warmup_return_to_dict(data, data_warmup, "sample_stats")

    @requires("prior")
    def sample_stats_prior_to_xarray(self):
        """Extract sample_stats from prior fit."""
        data, data_warmup = self.stats_to_xarray(self.prior)
        return self._warmup_return_to_dict(data, data_warmup, "sample_stats_prior")

    def stats_to_xarray(self, fit):
        """Extract sample_stats from fit."""
        dtypes = {
            "divergent__": bool,
            "n_leapfrog__": np.int64,
            "treedepth__": np.int64,
            **self.dtypes,
        }
        items = list(fit.method_variables().keys())
        rename_dict = {
            "divergent": "diverging",
            "n_leapfrog": "n_steps",
            "treedepth": "tree_depth",
            "stepsize": "step_size",
            "accept_stat": "acceptance_rate",
        }

        data, data_warmup = _unpack_fit(
            fit,
            items,
            self.save_warmup,
            self.dtypes,
        )
        for item in items:
            name = re.sub("__$", "", item)
            name = rename_dict.get(name, name)
            data[name] = data.pop(item).astype(dtypes.get(item, float))
            if data_warmup:
                data_warmup[name] = data_warmup.pop(item).astype(dtypes.get(item, float))
        return (data, data_warmup)

    @requires("posterior")
    @requires("posterior_predictive")
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        data, data_warmup = self.predictive_to_xarray(self.posterior_predictive, self.posterior)
        return self._warmup_return_to_dict(data, data_warmup, "posterior_predictive")

    @requires("prior")
    @requires("prior_predictive")
    def prior_predictive_to_xarray(self):
        """Convert prior_predictive samples to xarray."""
        data, data_warmup = self.predictive_to_xarray(self.prior_predictive, self.prior)
        return self._warmup_return_to_dict(data, data_warmup, "prior_predictive")

    def predictive_to_xarray(self, names, fit):
        """Convert predictive samples to xarray."""
        predictive = _as_set(names)

        data, data_warmup = _unpack_fit(
            fit,
            predictive,
            self.save_warmup,
            self.dtypes,
        )

        return (data, data_warmup)

    @requires("posterior")
    @requires("predictions")
    def predictions_to_xarray(self):
        """Convert out of sample predictions samples to xarray."""
        predictions = _as_set(self.predictions)

        data, data_warmup = _unpack_fit(
            self.posterior,
            predictions,
            self.save_warmup,
            self.dtypes,
        )

        return self._warmup_return_to_dict(data, data_warmup, "predictions")

    @requires("posterior")
    @requires("log_likelihood")
    def log_likelihood_to_xarray(self):
        """Convert elementwise log likelihood samples to xarray."""
        log_likelihood = _as_set(self.log_likelihood)

        data, data_warmup = _unpack_fit(
            self.posterior,
            log_likelihood,
            self.save_warmup,
            self.dtypes,
        )

        if isinstance(self.log_likelihood, dict):
            data = {obs_name: data[lik_name] for obs_name, lik_name in self.log_likelihood.items()}
            if data_warmup:
                data_warmup = {
                    obs_name: data_warmup[lik_name]
                    for obs_name, lik_name in self.log_likelihood.items()
                }
        return self._warmup_return_to_dict(data, data_warmup, "log_likelihood")

    @requires("prior")
    def prior_to_xarray(self):
        """Convert prior samples to xarray."""
        items = list(self.prior.stan_variables().keys())
        if self.prior_predictive is not None:
            try:
                items = _filter(items, self.prior_predictive)
            except ValueError:
                pass
        data, data_warmup = _unpack_fit(
            self.prior,
            items,
            self.save_warmup,
            self.dtypes,
        )

        return self._warmup_return_to_dict(data, data_warmup, "prior")

    @requires("observed_data")
    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        return dict_to_dataset(
            self.observed_data,
            inference_library=self.cmdstanpy,
            coords=self.coords,
            dims=self.dims,
            sample_dims=[],
            index_origin=self.index_origin,
        )

    @requires("constant_data")
    def constant_data_to_xarray(self):
        """Convert constant data to xarray."""
        return dict_to_dataset(
            self.constant_data,
            inference_library=self.cmdstanpy,
            coords=self.coords,
            dims=self.dims,
            sample_dims=[],
            index_origin=self.index_origin,
        )

    @requires("predictions_constant_data")
    def predictions_constant_data_to_xarray(self):
        """Convert constant data to xarray."""
        return dict_to_dataset(
            self.predictions_constant_data,
            inference_library=self.cmdstanpy,
            coords=self.coords,
            dims=self.dims,
            sample_dims=[],
            index_origin=self.index_origin,
        )

    def to_datatree(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created (i.e., there is no `output`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
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


def _as_set(spec):
    """Uniform representation for args which be name or list of names."""
    if spec is None:
        return []
    if isinstance(spec, str):
        return [spec]
    try:
        return set(spec.values())
    except AttributeError:
        return set(spec)


def _filter(names, spec):
    """Remove names from list of names."""
    if isinstance(spec, str):
        names.remove(spec)
    elif isinstance(spec, list):
        for item in spec:
            names.remove(item)
    elif isinstance(spec, dict):
        for item in spec.values():
            names.remove(item)
    return names


def _unpack_fit(fit, items, save_warmup, dtypes):
    """Transform fit to dictionary containing ndarrays.

    Parameters
    ----------
    data: cmdstanpy.CmdStanMCMC
    items: list
    save_warmup: bool
    dtypes: dict

    Returns
    -------
    dict
        key, values pairs. Values are formatted to shape = (chains, draws, *shape)
    """
    num_warmup = 0
    if save_warmup:
        if not fit._save_warmup:  # pylint: disable=protected-access
            save_warmup = False
        else:
            num_warmup = fit.num_draws_warmup

    nchains = fit.chains
    sample = {}
    sample_warmup = {}

    stan_variables = set(fit.stan_variables())
    method_variables = fit.method_variables()
    for item in items:
        if item in stan_variables:
            raw_draws = fit.stan_variable(item, inc_warmup=save_warmup)
            raw_draws = np.swapaxes(
                raw_draws.reshape((-1, nchains, *raw_draws.shape[1:]), order="F"), 0, 1
            )
        elif item in method_variables:
            raw_draws = np.swapaxes(method_variables[item].reshape((-1, nchains), order="F"), 0, 1)
        else:
            raise ValueError(f"fit data, unknown variable: {item}")
        raw_draws = raw_draws.astype(dtypes.get(item))
        if save_warmup:
            if item in method_variables:
                sample[item] = raw_draws
            else:
                sample_warmup[item] = raw_draws[:, :num_warmup, ...]
                sample[item] = raw_draws[:, num_warmup:, ...]
        else:
            sample[item] = raw_draws

    return sample, sample_warmup


def from_cmdstanpy(
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
    index_origin=None,
    coords=None,
    dims=None,
    save_warmup=None,
    dtypes=None,
):
    """Convert CmdStanPy data into an InferenceData object.

    For a usage example read the
    :ref:`Creating InferenceData section on from_cmdstanpy <creating_InferenceData>`

    Parameters
    ----------
    posterior : CmdStanMCMC object
        CmdStanPy CmdStanMCMC
    posterior_predictive : str, list of str
        Posterior predictive samples for the fit.
    predictions : str, list of str
        Out of sample prediction samples for the fit.
    prior : CmdStanMCMC
        CmdStanPy CmdStanMCMC
    prior_predictive : str, list of str
        Prior predictive samples for the fit.
    observed_data : dict
        Observed data used in the sampling.
    constant_data : dict
        Constant data used in the sampling.
    predictions_constant_data : dict
        Constant data for predictions used in the sampling.
    log_likelihood : str, list of str, dict of {str: str}, optional
        Pointwise log_likelihood for the data. If a dict, its keys should represent var_names
        from the corresponding observed data and its values the stan variable where the
        data is stored. By default, if a variable ``log_lik`` is present in the Stan model,
        it will be retrieved as pointwise log likelihood values. Use ``False``
        or set ``data.log_likelihood`` to false to avoid this behaviour.
    index_origin : int, optional
        Starting value of integer coordinate values. Defaults to the value in rcParam
        ``data.index_origin``.
    coords : dict of str or dict of iterable
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict of str or list of str
        A mapping from variables to a list of coordinate names for the variable.
    save_warmup : bool
        Save warmup iterations into InferenceData object, if found in the input files.
        If not defined, use default defined by the rcParams.
    dtypes: dict or str or cmdstanpy.CmdStanModel
        A dictionary containing dtype information (int, float) for parameters.
        If input is a string, it is assumed to be a model code or path to model code file.
        Model code can extracted from cmdstanpy.CmdStanModel object.

    Returns
    -------
    InferenceData object
    """
    return CmdStanPyConverter(
        posterior=posterior,
        posterior_predictive=posterior_predictive,
        predictions=predictions,
        prior=prior,
        prior_predictive=prior_predictive,
        observed_data=observed_data,
        constant_data=constant_data,
        predictions_constant_data=predictions_constant_data,
        log_likelihood=log_likelihood,
        index_origin=index_origin,
        coords=coords,
        dims=dims,
        save_warmup=save_warmup,
        dtypes=dtypes,
    ).to_datatree()
