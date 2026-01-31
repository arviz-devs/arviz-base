"""NumPyro-specific conversion code."""

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from xarray import DataTree

from arviz_base.base import dict_to_dataset, requires
from arviz_base.rcparams import rc_context, rcParams
from arviz_base.utils import expand_dims

# Module-level cache for lazy imports of optional dependencies
_IMPORT_CACHE = {}


def _get_jax():
    """Lazy import jax with caching."""
    if "jax" not in _IMPORT_CACHE:
        import jax

        _IMPORT_CACHE["jax"] = jax
    return _IMPORT_CACHE["jax"]


def _get_numpyro():
    """Lazy import numpyro with caching."""
    if "numpyro" not in _IMPORT_CACHE:
        import numpyro

        _IMPORT_CACHE["numpyro"] = numpyro
    return _IMPORT_CACHE["numpyro"]


class NumPyroInferenceAdapter(ABC):
    """Standardize methods across NumPyro inference objects for use with NumPyroConverter."""

    def __init__(self, inference_obj, model, model_args, model_kwargs, sample_shape):
        """Initialize the adapter with common attributes for NumPyro inference objects.

        This base class constructor sets up the shared infrastructure needed by all
        NumPyro inference adapters (MCMC, SVI, etc.) to provide a unified interface
        for the NumPyroConverter.

        Parameters
        ----------
        inference_obj : Any
            The NumPyro inference object to adapt (e.g., MCMC, SVI, or other inference types).
        model : callable
            The NumPyro model function that was used for inference.
        model_args : tuple, optional
            Positional arguments passed to the model during inference.
            If None, defaults to an empty tuple.
        model_kwargs : dict, optional
            Keyword arguments passed to the model during inference.
            If None, defaults to an empty dict.
        sample_shape : tuple of int
            Shape of the samples to be returned by get_samples().
            For MCMC: (num_chains, num_draws)
            For SVI: (num_samples,)
        """
        self.posterior = inference_obj
        self.model = model
        self._args = model_args or tuple()
        self._kwargs = model_kwargs or dict()
        self.sample_shape = sample_shape

        jax = _get_jax()
        self.numpyro = _get_numpyro()
        self.prng_key_func = jax.random.PRNGKey

    @property
    @abstractmethod
    def sample_dims(self):
        """Return the sample dimension names.

        Returns
        -------
        list of str
            Sample dimension names (e.g., ["chain", "draw"] for MCMC, ["sample"] for SVI).
        """
        raise NotImplementedError

    @abstractmethod
    def get_samples(self):
        """Get posterior samples from the inference object.

        Returns
        -------
        dict of {str: array-like}
            Dictionary mapping parameter names to their sampled values.
        """
        raise NotImplementedError

    def get_extra_fields(self, **kwargs):
        """Get extra fields from the inference object (e.g., divergences for MCMC).

        Returns
        -------
        dict of {str: array-like}
            Dictionary of extra diagnostic fields. Empty dict by default.
        """
        return dict()


class SVIAdapter(NumPyroInferenceAdapter):
    """Adapter for SVI to standardize attributes and methods with other inference objects."""

    def __init__(
        self,
        svi,
        *,
        svi_result,
        model_args=None,
        model_kwargs=None,
        num_samples: int = 1000,
    ):
        """Initialize SVI adapter for variational inference results.

        Parameters
        ----------
        svi : numpyro.infer.SVI
            Fitted SVI object.
        svi_result : numpyro.infer.svi.SVIRunResult
            SVI optimization results containing learned parameters.
        model_args : tuple, optional
            Positional arguments for the model.
        model_kwargs : dict, optional
            Keyword arguments for the model.
        num_samples : int, default 1000
            Number of posterior samples to generate from the guide.
        """
        if svi is None:
            raise ValueError("svi parameter is required for SVIAdapter")
        if svi_result is None:
            raise ValueError("svi_result parameter is required for SVIAdapter")

        super().__init__(
            svi,
            model=getattr(svi.guide, "model", svi.model),
            model_args=model_args,
            model_kwargs=model_kwargs,
            sample_shape=(num_samples,),
        )
        self.result_obj = svi_result

    @property
    def sample_dims(self):
        """Return sample dimension names for SVI.

        Returns
        -------
        list of str
            Sample dimension names for SVI: ["sample"].
        """
        return ["sample"]

    def get_samples(self, seed=None, group_by_chain=False, **kwargs):
        """Generate samples from SVI guide. Note: group_by_chain is ignored.

        Parameters
        ----------
        seed : int, optional
            Random seed for sampling.
        group_by_chain : bool, default False
            Ignored for SVI (included for API compatibility).
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict of {str: array-like}
            Dictionary mapping parameter names to their sampled values.
        """
        key = self.prng_key_func(seed or 0)
        if isinstance(self.posterior.guide, self.numpyro.infer.autoguide.AutoGuide):
            return self.posterior.guide.sample_posterior(
                key,
                self.result_obj.params,
                *self._args,
                sample_shape=self.sample_shape,
                **self._kwargs,
            )
        # if a custom guide is provided, sample by hand
        predictive = self.numpyro.infer.Predictive(
            self.posterior.guide, params=self.result_obj.params, num_samples=self.sample_shape[0]
        )
        samples = predictive(key, *self._args, **self._kwargs)
        return samples


class MCMCAdapter(NumPyroInferenceAdapter):
    """Adapter for MCMC to standardize attributes and methods with other inference objects."""

    def __init__(self, mcmc):
        """Initialize MCMC adapter from fitted MCMC object.

        Parameters
        ----------
        mcmc : numpyro.infer.MCMC
            Fitted MCMC object with completed sampling.
        """
        self.nchains = mcmc.num_chains
        self.ndraws = mcmc.num_samples // mcmc.thinning
        super().__init__(
            mcmc,
            model=mcmc.sampler.model,
            model_args=mcmc._args,
            model_kwargs=mcmc._kwargs,
            sample_shape=(self.nchains, self.ndraws),
        )

    @property
    def sample_dims(self):
        """Return sample dimension names for MCMC.

        Returns
        -------
        list of str
            Sample dimension names for MCMC: ["chain", "draw"].
        """
        return ["chain", "draw"]

    def get_samples(self, seed=None, group_by_chain=True, **kwargs):
        """Get MCMC samples. Note: seed is ignored.

        Parameters
        ----------
        seed : int, optional
            Ignored for MCMC (included for API compatibility).
        group_by_chain : bool, default True
            Whether to group samples by chain.
        **kwargs : dict
            Additional keyword arguments passed to MCMC.get_samples().

        Returns
        -------
        dict of {str: array-like}
            Dictionary mapping parameter names to their sampled values.
        """
        return self.posterior.get_samples(group_by_chain=group_by_chain, **kwargs)

    def get_extra_fields(self, **kwargs):
        """Get MCMC diagnostic fields (e.g., divergences, energy).

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to MCMC.get_extra_fields().

        Returns
        -------
        dict of {str: array-like}
            Dictionary of extra diagnostic fields from MCMC sampling.
        """
        return self.posterior.get_extra_fields(group_by_chain=True, **kwargs)


class NestedSamplerAdapter(NumPyroInferenceAdapter):
    """Adapter for NestedSampler to standardize attributes and methods."""

    def __init__(
        self,
        nested_sampler,
        *,
        model_args=None,
        model_kwargs=None,
        num_samples=1000,
    ):
        """Initialize NestedSampler adapter from fitted NestedSampler object.

        Parameters
        ----------
        nested_sampler : numpyro.contrib.nested_sampling.NestedSampler
            Fitted NestedSampler instance. Must have already called run() to generate results.
        model_args : tuple, optional
            Positional arguments for the model.
        model_kwargs : dict, optional
            Keyword arguments for the model.
        num_samples : int, default 1000
            Number of posterior samples to draw from the nested sampler results.
        """
        if nested_sampler is None:
            raise ValueError("nested_sampler parameter is required for NestedSamplerAdapter")

        super().__init__(
            nested_sampler,
            model=nested_sampler.model,
            model_args=model_args,
            model_kwargs=model_kwargs,
            sample_shape=(num_samples,),
        )
        self.num_samples = num_samples

    @property
    def sample_dims(self):
        """Return sample dimension names for NestedSampler.

        Returns
        -------
        list of str
            Sample dimension names for NestedSampler: ["sample"].
        """
        return ["sample"]

    def get_samples(self, seed=None, group_by_chain=False, **kwargs):
        """Get posterior samples from NestedSampler. Note: group_by_chain is ignored.

        Parameters
        ----------
        seed : int, optional
            Random seed for sampling.
        group_by_chain : bool, default False
            Ignored for NestedSampler (included for API compatibility).
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict of {str: array-like}
            Dictionary mapping parameter names to their sampled values.
        """
        key = self.prng_key_func(seed or 0)
        return self.posterior.get_samples(key, self.num_samples)


def _add_dims(dims_a, dims_b):
    """Merge two dimension mappings by concatenating dimension labels.

    Used to combine batch dims with event dims by appending the dims of dims_b to dims_a.

    Parameters
    ----------
    dims_a : dict of {str: list of str(s)}
        Mapping from site name to a list of dimension labels, typically
        representing batch dimensions.
    dims_b : dict of {str: list of str(s)}
        Mapping from site name to a list of dimension labels, typically
        representing event dimensions.

    Returns
    -------
    dict of {str: list of str(s)}
        Combined mapping where each site name is associated with the
        concatenated dimension labels from both inputs.
    """
    merged = defaultdict(list, dims_a)
    for k, v in dims_b.items():
        merged[k].extend(v)

    # Convert back to a regular dict
    return dict(merged)


def infer_dims(
    model,
    model_args=None,
    model_kwargs=None,
):
    """Infers batch dim names from numpyro model plates.

    Parameters
    ----------
    model : callable
        A numpyro model function.
    model_args : tuple of (Any, ...), optional
        Input args for the numpyro model.
    model_kwargs : dict of {str: Any}, optional
        Input kwargs for the numpyro model.

    Returns
    -------
    dict of {str: list of str(s)}
        Mapping from model site name to list of dimension labels.
    """
    jax = _get_jax()
    numpyro = _get_numpyro()
    dist = numpyro.distributions
    handlers = numpyro.handlers
    init_to_sample = numpyro.infer.initialization.init_to_sample
    PytreeTrace = numpyro.ops.pytree.PytreeTrace

    model_args = tuple() if model_args is None else model_args
    model_kwargs = dict() if model_kwargs is None else model_kwargs

    def _get_dist_name(fn):
        if isinstance(fn, dist.Independent | dist.ExpandedDistribution | dist.MaskedDistribution):
            return _get_dist_name(fn.base_dist)
        return type(fn).__name__

    def get_trace():
        # We use `init_to_sample` to get around ImproperUniform distribution,
        # which does not have `sample` method.
        subs_model = handlers.substitute(
            handlers.seed(model, 0),
            substitute_fn=init_to_sample,
        )
        trace = handlers.trace(subs_model).get_trace(*model_args, **model_kwargs)
        # Work around an issue where jax.eval_shape does not work
        # for distribution output (e.g. the function `lambda: dist.Normal(0, 1)`)
        # Here we will remove `fn` and store its name in the trace.
        for _, site in trace.items():
            if site["type"] == "sample":
                site["fn_name"] = _get_dist_name(site.pop("fn"))
            elif site["type"] == "deterministic":
                site["fn_name"] = "Deterministic"
        return PytreeTrace(trace)

    # We use eval_shape to avoid any array computation.
    trace = jax.eval_shape(get_trace).trace

    named_dims = {}

    # loop through the trace and pull the batch dim and event dim names
    for name, site in trace.items():
        batch_dims = [frame.name for frame in sorted(site["cond_indep_stack"], key=lambda x: x.dim)]
        event_dims = list(site.get("infer", {}).get("event_dims", []))

        # save the dim names leading with batch dims
        if site["type"] in ["sample", "deterministic"] and (batch_dims or event_dims):
            named_dims[name] = batch_dims + event_dims

    return named_dims


class NumPyroConverter:
    """Encapsulate NumPyro specific logic."""

    # pylint: disable=too-many-instance-attributes

    model = None

    def __init__(
        self,
        *,
        posterior=None,
        prior=None,
        posterior_predictive=None,
        predictions=None,
        constant_data=None,
        predictions_constant_data=None,
        log_likelihood=False,
        index_origin=None,
        coords=None,
        dims=None,
        pred_dims=None,
        extra_event_dims=None,
        num_chains=None,
    ):
        """Convert NumPyro data into an InferenceData object.

        Parameters
        ----------
        posterior : NumPyroInferenceAdapter
            A NumPyroInferenceAdapter child class
        prior : dict, optional
            Prior samples from a NumPyro model
        posterior_predictive : dict, optional
            Posterior predictive samples for the posterior
        predictions : dict, optional
            Out of sample predictions
        constant_data : dict, optional
            Dictionary containing constant data variables mapped to their values.
        predictions_constant_data : dict, optional
            Constant data used for out-of-sample predictions.
        log_likelihood : bool, default False
            Whether to compute and include log likelihood in the output.
        index_origin : int, optional
        coords : dict, optional
            Map of dimensions to coordinates
        dims : dict of {str : list of str}, optional
            Map variable names to their coordinates. Will be inferred if they are not provided.
        pred_dims : dict, optional
            Dims for predictions data. Map variable names to their coordinates.
        extra_event_dims : dict, optional
            Maps event dims that couldnt be inferred (ie deterministic sites) to their coordinates.
        num_chains : int, optional
            Number of chains used for sampling MCMC. Ignored if posterior is present, or if
            inference method is not MCMC.
        """
        jax = _get_jax()
        numpyro = _get_numpyro()

        self.posterior = posterior
        self.prior = jax.device_get(prior)
        self.posterior_predictive = jax.device_get(posterior_predictive)
        self.predictions = predictions
        self.constant_data = constant_data
        self.predictions_constant_data = predictions_constant_data
        self.log_likelihood = log_likelihood
        self.index_origin = rcParams["data.index_origin"] if index_origin is None else index_origin
        self.coords = coords
        self.dims = dims
        self.pred_dims = pred_dims
        self.extra_event_dims = extra_event_dims
        self.numpyro = numpyro

        # use nchains to help infer shape when posterior isnt present for MCMC
        self.nchains = (num_chains or 1) if rcParams["data.sample_dims"][0] == "chain" else None

        if posterior is not None:
            samples = jax.device_get(self.posterior.get_samples())
            if hasattr(samples, "_asdict"):
                # In case it is easy to convert to a dictionary, as in the case of namedtuples
                samples = {k: expand_dims(v) for k, v in samples._asdict().items()}
            if not isinstance(samples, dict):
                # handle the case we run MCMC with a general potential_fn
                # (instead of a NumPyro model) whose args is not a dictionary
                # (e.g. f(x) = x ** 2)
                tree_flatten_samples = jax.tree_util.tree_flatten(samples)[0]
                samples = {
                    f"Param:{i}": jax.device_get(v) for i, v in enumerate(tree_flatten_samples)
                }
            self._samples = samples
            self.model = self.posterior.model
            self.sample_shape = self.posterior.sample_shape

            # model arguments and keyword arguments
            self._args = self.posterior._args  # pylint: disable=protected-access
            self._kwargs = self.posterior._kwargs  # pylint: disable=protected-access
            self.dims = self.dims if self.dims is not None else self.infer_dims()
            self.pred_dims = (
                self.pred_dims if self.pred_dims is not None else self.infer_pred_dims()
            )
        else:
            self.sample_shape = self._infer_sample_shape()

        observations = {}
        if self.model is not None:
            trace = self._get_model_trace(
                self.model,
                model_args=self._args,
                model_kwargs=self._kwargs,
                key=jax.random.PRNGKey(0),
            )
            observations = {
                name: site["value"]
                for name, site in trace.items()
                if site["type"] == "sample" and site["is_observed"]
            }
        self.observations = observations if observations else None

    def _get_model_trace(self, model, model_args, model_kwargs, key):
        """Extract the numpyro model trace."""
        model_args = model_args or tuple()
        model_kwargs = model_kwargs or dict()

        # we need to use an init strategy to generate random samples for ImproperUniform sites
        seeded_model = self.numpyro.handlers.substitute(
            self.numpyro.handlers.seed(model, key),
            substitute_fn=self.numpyro.infer.init_to_sample,
        )
        trace = self.numpyro.handlers.trace(seeded_model).get_trace(*model_args, **model_kwargs)
        return trace

    def _infer_sample_shape(self):
        # try to use these sources to infer sample shape
        sources = [
            self.predictions,
            self.posterior_predictive,
            self.prior,
        ]
        # pick first available source
        get_from = next((src for src in sources if src is not None), None)
        no_constant_data = self.constant_data is None and self.predictions_constant_data is None
        if get_from is not None:
            aelem = next(iter(get_from.values()))  # pick an arbitrary element

            # For MCMC from numpyro, we need to reshape the sample shape
            # based on the number of chains provided
            if self.nchains is not None:
                ndraws = aelem.shape[0] // self.nchains
                return (self.nchains, ndraws)
            else:
                return aelem.shape[: len(rcParams["data.sample_dims"])]
        elif no_constant_data:
            raise ValueError(
                "When constructing InferenceData, must have at least one of "
                "posterior, prior, posterior_predictive, or predictions."
            )
        else:
            # fallback shape when theres no inference, but there is constant data
            return (1,) * len(rcParams["data.sample_dims"])

    @requires("posterior")
    def posterior_to_xarray(self):
        """Convert the posterior to an xarray dataset."""
        data = self._samples
        return dict_to_dataset(
            data,
            inference_library=self.numpyro,
            coords=self.coords,
            dims=self.dims,
            index_origin=self.index_origin,
        )

    @requires("posterior")
    def sample_stats_to_xarray(self):
        """Extract sample_stats from NumPyro posterior."""
        rename_key = {
            "potential_energy": "lp",
            "adapt_state.step_size": "step_size",
            "num_steps": "n_steps",
            "accept_prob": "acceptance_rate",
        }
        data = {}
        for stat, value in self.posterior.get_extra_fields().items():
            if isinstance(value, dict | tuple):
                continue
            name = rename_key.get(stat, stat)
            value_cp = value.copy()
            data[name] = value_cp
            if stat == "num_steps":
                data["tree_depth"] = np.log2(value_cp).astype(int) + 1

        return dict_to_dataset(
            data,
            inference_library=self.numpyro,
            dims=None,
            coords=self.coords,
            index_origin=self.index_origin,
        )

    @requires("posterior")
    @requires("model")
    def log_likelihood_to_xarray(self):
        """Extract log likelihood from NumPyro posterior."""
        if not self.log_likelihood:
            return None
        data = {}
        if self.observations is not None:
            samples = self.posterior.get_samples(group_by_chain=False)
            if hasattr(samples, "_asdict"):
                samples = samples._asdict()
            log_likelihood_dict = self.numpyro.infer.log_likelihood(
                self.model, samples, *self._args, **self._kwargs
            )
            for obs_name, log_like in log_likelihood_dict.items():
                shape = self.sample_shape + log_like.shape[1:]
                data[obs_name] = np.reshape(np.asarray(log_like), shape)
        return dict_to_dataset(
            data,
            inference_library=self.numpyro,
            dims=self.dims,
            coords=self.coords,
            index_origin=self.index_origin,
            skip_event_dims=True,
        )

    def translate_posterior_predictive_dict_to_xarray(self, dct, dims):
        """Convert posterior_predictive or prediction samples to xarray."""
        data = {}
        for k, ary in dct.items():
            shape = ary.shape
            if (shape[: len(self.sample_shape)] == self.sample_shape) or shape[0] == np.prod(
                self.sample_shape
            ):
                data[k] = ary.reshape(self.sample_shape + shape[1:])
            else:
                data[k] = expand_dims(ary)
                warnings.warn(
                    "posterior predictive shape not compatible with number of chains and draws. "
                    "This can mean that some draws or even whole chains are not represented."
                )
        return dict_to_dataset(
            data,
            inference_library=self.numpyro,
            coords=self.coords,
            dims=dims,
            index_origin=self.index_origin,
        )

    @requires("posterior_predictive")
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        return self.translate_posterior_predictive_dict_to_xarray(
            self.posterior_predictive, self.dims
        )

    @requires("predictions")
    def predictions_to_xarray(self):
        """Convert predictions to xarray."""
        return self.translate_posterior_predictive_dict_to_xarray(self.predictions, self.pred_dims)

    def priors_to_xarray(self):
        """Convert prior samples (and if possible prior predictive too) to xarray."""
        if self.prior is None:
            return {"prior": None, "prior_predictive": None}
        if self.posterior is not None:
            prior_vars = list(self._samples.keys())
            prior_predictive_vars = [key for key in self.prior.keys() if key not in prior_vars]
        else:
            prior_vars = self.prior.keys()
            prior_predictive_vars = None

        # dont expand dims for SVI
        expand_dims_func = expand_dims if len(rcParams["data.sample_dims"]) > 1 else lambda x: x
        priors_dict = {
            group: (
                None
                if var_names is None
                else dict_to_dataset(
                    {k: expand_dims_func(self.prior[k]) for k in var_names},
                    inference_library=self.numpyro,
                    coords=self.coords,
                    dims=self.dims,
                    index_origin=self.index_origin,
                )
            )
            for group, var_names in zip(
                ("prior", "prior_predictive"), (prior_vars, prior_predictive_vars)
            )
        }
        return priors_dict

    @requires("observations")
    @requires("model")
    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        return dict_to_dataset(
            self.observations,
            inference_library=self.numpyro,
            dims=self.dims,
            coords=self.coords,
            sample_dims=[],
            index_origin=self.index_origin,
        )

    @requires("constant_data")
    def constant_data_to_xarray(self):
        """Convert constant_data to xarray."""
        return dict_to_dataset(
            self.constant_data,
            inference_library=self.numpyro,
            dims=self.dims,
            coords=self.coords,
            sample_dims=[],
            index_origin=self.index_origin,
        )

    @requires("predictions_constant_data")
    def predictions_constant_data_to_xarray(self):
        """Convert predictions_constant_data to xarray."""
        return dict_to_dataset(
            self.predictions_constant_data,
            inference_library=self.numpyro,
            dims=self.pred_dims,
            coords=self.coords,
            sample_dims=[],
            index_origin=self.index_origin,
        )

    def to_datatree(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created (i.e., there is no `trace`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        """
        dicto = {
            "posterior": self.posterior_to_xarray(),
            "sample_stats": self.sample_stats_to_xarray(),
            "log_likelihood": self.log_likelihood_to_xarray(),
            "posterior_predictive": self.posterior_predictive_to_xarray(),
            "predictions": self.predictions_to_xarray(),
            **self.priors_to_xarray(),
            "observed_data": self.observed_data_to_xarray(),
            "constant_data": self.constant_data_to_xarray(),
            "predictions_constant_data": self.predictions_constant_data_to_xarray(),
        }

        return DataTree.from_dict({group: ds for group, ds in dicto.items() if ds is not None})

    @requires("posterior")
    @requires("model")
    def infer_dims(self) -> dict[str, list[str]]:
        """Infers dims for input data."""
        dims = infer_dims(self.model, self._args, self._kwargs)
        if self.extra_event_dims:
            dims = _add_dims(dims, self.extra_event_dims)
        return dims

    @requires("posterior")
    @requires("model")
    @requires("predictions")
    def infer_pred_dims(self) -> dict[str, list[str]]:
        """Infers dims for predictions data."""
        dims = infer_dims(self.model, self._args, self._kwargs)
        if self.extra_event_dims:
            dims = _add_dims(dims, self.extra_event_dims)
        return dims


def from_numpyro(
    posterior=None,
    *,
    prior=None,
    posterior_predictive=None,
    predictions=None,
    constant_data=None,
    predictions_constant_data=None,
    log_likelihood=False,
    index_origin=None,
    coords=None,
    dims=None,
    pred_dims=None,
    extra_event_dims=None,
    num_chains=1,
):
    """Convert NumPyro mcmc inference data into a DataTree object.

    For a usage example read :ref:`numpyro_conversion`

    If no dims are provided, this will infer batch dim names from NumPyro model plates.
    For event dim names, such as with the ZeroSumNormal, `infer={"event_dims":dim_names}`
    can be provided in numpyro.sample, i.e.::

        # equivalent to dims entry, {"gamma": ["groups"]}
        gamma = numpyro.sample(
            "gamma",
            dist.ZeroSumNormal(1, event_shape=(n_groups,)),
            infer={"event_dims":["groups"]}
        )

    There is also an additional `extra_event_dims` input to cover any edge cases, for instance
    deterministic sites with event dims (which dont have an `infer` argument to provide metadata).

    Parameters
    ----------
    posterior : numpyro.infer.MCMC
        Fitted MCMC object from NumPyro
    prior : dict, optional
        Prior samples from a NumPyro model
    posterior_predictive : dict, optional
        Posterior predictive samples for the posterior
    predictions : dict, optional
        Out of sample predictions
    constant_data : dict, optional
        Dictionary containing constant data variables mapped to their values.
    predictions_constant_data : dict, optional
        Constant data used for out-of-sample predictions.
    log_likelihood : bool, default False
        Whether to compute and include log likelihood in the output.
    index_origin : int, optional
    coords : dict, optional
        Map of dimensions to coordinates
    dims : dict of {str : list of str}, optional
        Map variable names to their coordinates. Will be inferred if they are not provided.
    pred_dims : dict, optional
        Dims for predictions data. Map variable names to their coordinates. Default behavior is to
        infer dims if this is not provided
    extra_event_dims : dict, optional
        Extra event dims for deterministic sites. Maps event dims that couldnt be inferred to
        their coordinates.
    num_chains : int, default 1
        Number of chains used for sampling. Ignored if posterior is present.

    Returns
    -------
    DataTree
    """
    with rc_context(rc={"data.sample_dims": ["chain", "draw"]}):
        posterior = MCMCAdapter(posterior) if posterior is not None else None
        return NumPyroConverter(
            posterior=posterior,
            prior=prior,
            posterior_predictive=posterior_predictive,
            predictions=predictions,
            constant_data=constant_data,
            predictions_constant_data=predictions_constant_data,
            log_likelihood=log_likelihood,
            index_origin=index_origin,
            coords=coords,
            dims=dims,
            pred_dims=pred_dims,
            extra_event_dims=extra_event_dims,
            num_chains=num_chains,
        ).to_datatree()


def from_numpyro_svi(
    svi=None,
    *,
    svi_result=None,
    model_args=None,
    model_kwargs=None,
    prior=None,
    posterior_predictive=None,
    predictions=None,
    constant_data=None,
    predictions_constant_data=None,
    log_likelihood=False,
    index_origin=None,
    coords=None,
    dims=None,
    pred_dims=None,
    extra_event_dims=None,
    num_samples: int = 1000,
):
    """Convert NumPyro SVI results into a DataTree object.

    For a usage example read :ref:`numpyro_conversion`

    If no dims are provided, this will infer batch dim names from NumPyro model plates.
    For event dim names, such as with the ZeroSumNormal, `infer={"event_dims":dim_names}`
    can be provided in numpyro.sample, i.e.::

        # equivalent to dims entry, {"gamma": ["groups"]}
        gamma = numpyro.sample(
            "gamma",
            dist.ZeroSumNormal(1, event_shape=(n_groups,)),
            infer={"event_dims":["groups"]}
        )

    There is also an additional `extra_event_dims` input to cover any edge cases, for instance
    deterministic sites with event dims (which dont have an `infer` argument to provide metadata).

    Parameters
    ----------
    svi : numpyro.infer.SVI, optional
        Numpyro SVI instance used for fitting the model. If not provided, no posterior
        will be included in the output, and at least one of prior, posterior_predictive,
        or predictions must be provided.
    svi_result : numpyro.infer.svi.SVIRunResult, optional
        SVI results from a fitted model. Required if SVI is provided.
    model_args : tuple, optional
        Model arguments, should match those used for fitting the model.
    model_kwargs : dict, optional
        Model keyword arguments, should match those used for fitting the model.
    prior : dict, optional
        Prior samples from a NumPyro model
    posterior_predictive : dict, optional
        Posterior predictive samples for the posterior
    predictions : dict, optional
        Out of sample predictions
    constant_data : dict, optional
        Dictionary containing constant data variables mapped to their values.
    predictions_constant_data : dict, optional
        Constant data used for out-of-sample predictions.
    log_likelihood : bool, default False
        Whether to compute and include log likelihood in the output.
    index_origin : int, optional
    coords : dict, optional
        Map of dimensions to coordinates
    dims : dict of {str : list of str}, optional
        Map variable names to their coordinates. Will be inferred if they are not provided.
    pred_dims : dict, optional
        Dims for predictions data. Map variable names to their coordinates. Default behavior is to
        infer dims if this is not provided
    extra_event_dims : dict, optional
        Extra event dims for deterministic sites. Maps event dims that couldnt be inferred to
        their coordinates.
    num_samples : int, default 1000
        Number of posterior samples to generate

    Returns
    -------
    DataTree
    """
    with rc_context(rc={"data.sample_dims": ["sample"]}):
        posterior = (
            SVIAdapter(
                svi,
                svi_result=svi_result,
                model_args=model_args,
                model_kwargs=model_kwargs,
                num_samples=num_samples,
            )
            if svi is not None
            else None
        )
        return NumPyroConverter(
            posterior=posterior,
            prior=prior,
            posterior_predictive=posterior_predictive,
            predictions=predictions,
            constant_data=constant_data,
            predictions_constant_data=predictions_constant_data,
            log_likelihood=log_likelihood,
            index_origin=index_origin,
            coords=coords,
            dims=dims,
            pred_dims=pred_dims,
            extra_event_dims=extra_event_dims,
        ).to_datatree()


def from_numpyro_nested_sampler(
    nested_sampler=None,
    *,
    model_args=None,
    model_kwargs=None,
    prior=None,
    posterior_predictive=None,
    predictions=None,
    constant_data=None,
    predictions_constant_data=None,
    log_likelihood=False,
    index_origin=None,
    coords=None,
    dims=None,
    pred_dims=None,
    extra_event_dims=None,
    num_samples: int = 1000,
):
    """Convert NumPyro NestedSampler results into a DataTree object.

    For a usage example read :ref:`numpyro_conversion`

    If no dims are provided, this will infer batch dim names from NumPyro model plates.
    For event dim names, such as with the ZeroSumNormal, `infer={"event_dims":dim_names}`
    can be provided in numpyro.sample, i.e.::

        # equivalent to dims entry, {"gamma": ["groups"]}
        gamma = numpyro.sample(
            "gamma",
            dist.ZeroSumNormal(1, event_shape=(n_groups,)),
            infer={"event_dims":["groups"]}
        )

    There is also an additional `extra_event_dims` input to cover any edge cases, for instance
    deterministic sites with event dims (which dont have an `infer` argument to provide metadata).

    Parameters
    ----------
    nested_sampler : numpyro.contrib.nested_sampling.NestedSampler, optional
        Fitted NestedSampler instance. Must have already called run() to generate results.
        If not provided, no posterior will be included in the output, and at least one of
        prior, posterior_predictive, or predictions must be provided.
    model_args : tuple, optional
        Model arguments, should match those used for fitting the model.
    model_kwargs : dict, optional
        Model keyword arguments, should match those used for fitting the model.
    prior : dict, optional
        Prior samples from a NumPyro model
    posterior_predictive : dict, optional
        Posterior predictive samples for the posterior
    predictions : dict, optional
        Out of sample predictions
    constant_data : dict, optional
        Dictionary containing constant data variables mapped to their values.
    predictions_constant_data : dict, optional
        Constant data used for out-of-sample predictions.
    log_likelihood : bool, default False
        Whether to compute and include log likelihood in the output.
    index_origin : int, optional
    coords : dict, optional
        Map of dimensions to coordinates
    dims : dict of {str : list of str}, optional
        Map variable names to their coordinates. Will be inferred if they are not provided.
    pred_dims : dict, optional
        Dims for predictions data. Map variable names to their coordinates. Default behavior is to
        infer dims if this is not provided
    extra_event_dims : dict, optional
        Extra event dims for deterministic sites. Maps event dims that couldnt be inferred to
        their coordinates.
    num_samples : int, default 1000
        Number of posterior samples to generate from nested sampler results

    Returns
    -------
    DataTree
    """
    with rc_context(rc={"data.sample_dims": ["sample"]}):
        posterior = (
            NestedSamplerAdapter(
                nested_sampler,
                model_args=model_args,
                model_kwargs=model_kwargs,
                num_samples=num_samples,
            )
            if nested_sampler is not None
            else None
        )
        return NumPyroConverter(
            posterior=posterior,
            prior=prior,
            posterior_predictive=posterior_predictive,
            predictions=predictions,
            constant_data=constant_data,
            predictions_constant_data=predictions_constant_data,
            log_likelihood=log_likelihood,
            index_origin=index_origin,
            coords=coords,
            dims=dims,
            pred_dims=pred_dims,
            extra_event_dims=extra_event_dims,
        ).to_datatree()
