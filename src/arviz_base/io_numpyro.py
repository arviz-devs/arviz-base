"""NumPyro-specific conversion code."""

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

import lazy_loader as _lazy
import numpy as np
from numpy.typing import ArrayLike
from xarray import DataTree

from arviz_base.base import dict_to_dataset, requires
from arviz_base.rcparams import rc_context, rcParams
from arviz_base.utils import expand_dims

if TYPE_CHECKING:
    import jax
    import numpyro
else:
    jax = _lazy.load("jax")
    numpyro = _lazy.load("numpyro")


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
    def get_samples(self, seed=None, **kwargs):
        """Get posterior samples from the inference object.

        Parameters
        ----------
        seed : int, optional
            Random seed for sampling. Not all inference types use this parameter.
            For MCMC, this parameter is ignored as samples are already drawn.
            For SVI, this controls random number generation.
        **kwargs : dict
            Additional keyword arguments passed to the underlying inference object's
            sampling method.

        Returns
        -------
        dict of {str: array-like}
            Dictionary mapping parameter names to their sampled values.
            Arrays should have shape (*sample_dims, ...)
        """
        raise NotImplementedError

    def get_sample_stats(self, **kwargs):
        """Get sample stats from the inference object (e.g., divergences for MCMC).

        Returns
        -------
        dict of {str: array-like}
            Dictionary of sample stats. Empty dict by default.
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
    def sample_dims(self) -> list[str]:  # noqa: D102
        return ["sample"]

    def get_samples(  # noqa: D102
        self, seed: int | None = None, **kwargs: dict
    ) -> dict[str, ArrayLike]:
        key = self.prng_key_func(seed or 0)
        if isinstance(self.posterior.guide, numpyro.infer.autoguide.AutoGuide):
            return self.posterior.guide.sample_posterior(
                key,
                self.result_obj.params,
                *self._args,
                sample_shape=self.sample_shape,
                **self._kwargs,
            )
        # if a custom guide is provided, sample by hand
        predictive = numpyro.infer.Predictive(
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
        self._max_tree_depth = getattr(mcmc.sampler, "_max_tree_depth", None)
        super().__init__(
            mcmc,
            model=mcmc.sampler.model,
            model_args=mcmc._args,
            model_kwargs=mcmc._kwargs,
            sample_shape=(self.nchains, self.ndraws),
        )

    @property
    def sample_dims(self) -> list[str]:  # noqa: D102
        return ["chain", "draw"]

    def get_samples(  # noqa: D102
        self, seed: int | None = None, **kwargs: dict
    ) -> dict[str, ArrayLike]:
        return self.posterior.get_samples(group_by_chain=True, **kwargs)

    def get_sample_stats(self, **kwargs) -> dict[str, ArrayLike]:  # noqa: D102
        return self.posterior.get_extra_fields(group_by_chain=True, **kwargs)


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

        # use nchains to help infer shape when posterior isnt present for MCMC
        self.nchains = num_chains if rcParams["data.sample_dims"][0] == "chain" else None

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
        seeded_model = numpyro.handlers.substitute(
            numpyro.handlers.seed(model, key),
            substitute_fn=numpyro.infer.init_to_sample,
        )
        trace = numpyro.handlers.trace(seeded_model).get_trace(*model_args, **model_kwargs)
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
            aelem = next(iter(get_from.values()))
            batch_ndim = aelem.ndim - len(rcParams["data.sample_dims"])
            if batch_ndim == 0:
                if self.nchains is None:
                    warnings.warn(
                        f"Input has no extra dims beyond sample_dims "
                        f"{rcParams['data.sample_dims']}, but nchains=None. "
                        f"\nPlease check your `nchains` and `sample_dims` inputs."
                        f"\nFalling back to treating the first axis as the only sample dim, "
                        f"giving sample_shape={aelem.shape[:1]}."
                    )
                    return aelem.shape[:1]

                # nchains is known: split flat array into (nchains, ndraws)
                ndraws, remainder = divmod(aelem.shape[0], self.nchains)
                if remainder != 0:
                    raise ValueError(
                        f"Sample shape {aelem.shape} is not divisible "
                        f"by the number of chains {self.nchains}."
                    )
                return (self.nchains, ndraws)
            else:
                # Array already has chain/draw dims; optionally validate against nchains
                sample_shape = aelem.shape[: len(rcParams["data.sample_dims"])]
                if self.nchains is not None and sample_shape[0] != self.nchains:
                    raise ValueError(
                        f"Array shape {aelem.shape} implies {sample_shape[0]} chains, "
                        f"but nchains={self.nchains}."
                    )
                return sample_shape

        elif no_constant_data:
            raise ValueError(
                "When constructing InferenceData, must have at least one of "
                "posterior, prior, posterior_predictive, or predictions."
            )
        else:
            # fallback shape when there's no inference, but there is constant data
            fallback_shape = (
                (self.nchains, 1)
                if self.nchains is not None
                else (1,) * len(rcParams["data.sample_dims"])
            )
            warnings.warn(
                f"No posterior, prior, or predictive samples provided. "
                f"Defaulting to sample_shape={fallback_shape}. "
                f"This may cause unexpected behavior in downstream operations."
            )
            return fallback_shape

    @requires("posterior")
    def posterior_to_xarray(self):
        """Convert the posterior to an xarray dataset."""
        data = self._samples
        return dict_to_dataset(
            data,
            inference_library=numpyro,
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
        for stat, value in self.posterior.get_sample_stats().items():
            if isinstance(value, dict | tuple):
                continue
            name = rename_key.get(stat, stat)
            value_cp = value.copy()
            data[name] = value_cp
            if stat == "num_steps":
                data["tree_depth"] = np.log2(value_cp).astype(int) + 1
                if self.posterior._max_tree_depth is not None:
                    data["reached_max_tree_depth"] = (
                        data["tree_depth"] >= self.posterior._max_tree_depth
                    )

        return dict_to_dataset(
            data,
            inference_library=numpyro,
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
            samples = self.posterior.get_samples()
            data = numpyro.infer.log_likelihood(
                self.model,
                samples,
                *self._args,
                **self._kwargs,
                batch_ndims=len(rcParams["data.sample_dims"]),
            )
        return dict_to_dataset(
            data,
            inference_library=numpyro,
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
            if shape[: len(self.sample_shape)] == self.sample_shape:
                data[k] = ary
            elif shape[0] == np.prod(self.sample_shape):
                data[k] = ary.reshape(self.sample_shape + shape[1:])
            else:
                data[k] = expand_dims(ary)
                warnings.warn(
                    "posterior predictive shape not compatible with sample shape. "
                    "This can mean that some sample dims are not represented."
                )
        return dict_to_dataset(
            data,
            inference_library=numpyro,
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

        # priors input should have same ndims as sample_dims + batch_dims
        data = {}
        for k, ary in self.prior.items():
            ndims = len(ary.shape)
            batch_dims = 0
            if self.dims:
                batch_dims += len(self.dims.get(k, []))
            expected_ndims = len(rcParams["data.sample_dims"]) + batch_dims
            if ndims < expected_ndims:
                data[k] = expand_dims(ary)
            else:
                data[k] = ary

        priors_dict = {}
        for group, var_names in zip(
            ("prior", "prior_predictive"), (prior_vars, prior_predictive_vars)
        ):
            if var_names is None:
                priors_dict[group] = None
            else:
                filtered = {k: v for k, v in data.items() if k in var_names}
                priors_dict[group] = dict_to_dataset(
                    filtered,
                    inference_library=numpyro,
                    coords=self.coords,
                    dims=self.dims,
                    index_origin=self.index_origin,
                )
        return priors_dict

    @requires("observations")
    @requires("model")
    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        return dict_to_dataset(
            self.observations,
            inference_library=numpyro,
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
            inference_library=numpyro,
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
            inference_library=numpyro,
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
    sample_dims=None,
    num_chains=None,
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
    posterior : numpyro.infer.MCMC | NumPyroInferenceAdapter
        A fitted MCMC object from NumPyro, or an instance of a child class
        of NumPyroInferenceAdapter.
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
    sample_dims : list of str, optional
        Names of the sample dimensions (e.g., ["chain", "draw"] for MCMC, ["sample"] for SVI).
        Must be provided if `posterior` is None. If `posterior` is provided, this argument
        is ignored and overwritten with `posterior.sample_dims`.
    num_chains : int, optional
        Number of chains used for sampling. Defaults to 1 for MCMC if not provided.
        Ignored if posterior is present.

    Returns
    -------
    DataTree
    """
    if posterior is None:
        if sample_dims is None:
            raise ValueError(
                "sample_dims must be provided if posterior is None. "
                "For MCMC use ['chain', 'draw'], for SVI use ['sample']."
            )
    elif isinstance(posterior, numpyro.infer.MCMC):
        sample_dims = ["chain", "draw"]
        posterior = MCMCAdapter(posterior)
    else:
        sample_dims = posterior.sample_dims

    with rc_context(rc={"data.sample_dims": sample_dims}):
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
    num_samples=1000,
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
