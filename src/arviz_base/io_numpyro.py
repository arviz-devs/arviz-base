"""NumPyro-specific conversion code."""

import warnings
from collections import defaultdict

import numpy as np
from xarray import DataTree

from arviz_base.base import dict_to_dataset, requires
from arviz_base.rcparams import rc_context, rcParams
from arviz_base.utils import expand_dims


class SVIWrapper:
    """A helper class for SVI to mimic numpyro.infer.MCMC methods."""

    def __init__(
        self,
        svi,
        *,
        svi_result,
        model_args=None,
        model_kwargs=None,
        num_samples: int = 1000,
    ):
        import jax
        import numpyro

        self.svi = svi
        self.svi_result = svi_result
        self._args = model_args or tuple()
        self._kwargs = model_kwargs or dict()
        self.num_samples = num_samples
        self.thinning = 1
        self.num_chains = 0
        self.sample_dims = ["sample"]
        self.kind = "svi"

        self.numpyro = numpyro
        self.prng_key_func = jax.random.PRNGKey

    def get_samples(self, seed=None, **kwargs):
        """Mimics mcmc.get_samples()."""
        key = self.prng_key_func(seed or 0)
        if isinstance(self.svi.guide, self.numpyro.infer.autoguide.AutoGuide):
            return self.svi.guide.sample_posterior(
                key,
                self.svi_result.params,
                *self._args,
                sample_shape=(self.num_samples,),
                **self._kwargs,
            )
        # if a custom guide is provided, sample by hand
        predictive = self.numpyro.infer.Predictive(
            self.svi.guide, params=self.svi_result.params, num_samples=self.num_samples
        )
        samples = predictive(key, *self._args, **self._kwargs)
        return samples

    @property
    def sampler(self):
        """Mimics mcmc.sampler.model."""

        class Sampler:
            def __init__(self, model):
                self._model = model

            @property
            def model(self):
                return self._model

        return Sampler(getattr(self.svi.guide, "model", self.svi.model))

    def get_extra_fields(self, **kwargs):
        """Mimics mcmc.get_extra_fields()."""
        return dict()


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
    import jax
    from numpyro import distributions as dist
    from numpyro import handlers
    from numpyro.infer.initialization import init_to_sample
    from numpyro.ops.pytree import PytreeTrace

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
    nchains = None
    ndraws = None

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
        num_chains=1,
    ):
        """Convert NumPyro data into an InferenceData object.

        Parameters
        ----------
        posterior : numpyro.mcmc.MCMC
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
        index_origin : int, optional
        coords : dict, optional
            Map of dimensions to coordinates
        dims : dict of {str : list of str}, optional
            Map variable names to their coordinates. Will be inferred if they are not provided.
        pred_dims : dict, optional
            Dims for predictions data. Map variable names to their coordinates.
        num_chains : int, optional
            Number of chains used for sampling. Ignored if posterior is present.
        extra_event_dims : dict, optional
            Maps event dims that couldnt be inferred (ie deterministic sites) to their coordinates.
        """
        import jax
        import numpyro

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

        def arbitrary_element(dct):
            return next(iter(dct.values()))

        if posterior is not None:
            samples = jax.device_get(self.posterior.get_samples(group_by_chain=True))
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
            self.nchains, self.ndraws = (
                posterior.num_chains,
                posterior.num_samples // posterior.thinning,
            )
            self.model = self.posterior.sampler.model
            # model arguments and keyword arguments
            self._args = self.posterior._args  # pylint: disable=protected-access
            self._kwargs = self.posterior._kwargs  # pylint: disable=protected-access
            self.dims = self.dims if self.dims is not None else self.infer_dims()
            self.pred_dims = (
                self.pred_dims if self.pred_dims is not None else self.infer_pred_dims()
            )
        else:
            self.nchains = num_chains
            get_from = None
            if predictions is not None:
                get_from = predictions
            elif posterior_predictive is not None:
                get_from = posterior_predictive
            elif prior is not None:
                get_from = prior
            if get_from is None and constant_data is None and predictions_constant_data is None:
                raise ValueError(
                    "When constructing InferenceData must have at least"
                    " one of posterior, prior, posterior_predictive or predictions."
                )
            if get_from is not None:
                aelem = arbitrary_element(get_from)
                self.ndraws = aelem.shape[0] // self.nchains

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
        for stat, value in self.posterior.get_extra_fields(group_by_chain=True).items():
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
                shape = (self.nchains, self.ndraws) + log_like.shape[1:]
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
            if shape[0] == self.nchains and shape[1] == self.ndraws:
                data[k] = ary
            elif (
                shape[0] == self.nchains * self.ndraws
                and getattr(self.posterior, "kind", "") != "svi"
            ):
                data[k] = ary.reshape((self.nchains, self.ndraws, *shape[1:]))
            elif getattr(self.posterior, "kind", "") == "svi":
                data[k] = ary
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
        expand_dims_func = (
            expand_dims if getattr(self.posterior, "kind", "") != "svi" else lambda x: x
        )
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
    log_likelihood=None,
    index_origin=None,
    coords=None,
    dims=None,
    pred_dims=None,
    extra_event_dims=None,
    num_chains=1,
):
    """Convert NumPyro data into a DataTree object.

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
    posterior : numpyro.mcmc.MCMC
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
    svi,
    *,
    svi_result,
    model_args=None,
    model_kwargs=None,
    prior=None,
    posterior_predictive=None,
    predictions=None,
    constant_data=None,
    predictions_constant_data=None,
    log_likelihood=None,
    index_origin=None,
    coords=None,
    dims=None,
    pred_dims=None,
    extra_event_dims=None,
    model=None,
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
    svi : numpyro.infer.svi.SVI
        Numpyro SVI instance used for fitting the model.
    svi_result : numpyro.infer.svi.SVIRunResult
        SVI results from a fitted model.
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
    posterior = SVIWrapper(
        svi,
        svi_result=svi_result,
        model_args=model_args,
        model_kwargs=model_kwargs,
        num_samples=num_samples,
    )
    with rc_context(rc={"data.sample_dims": ["sample"]}):
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
            num_chains=0,
        ).to_datatree()
