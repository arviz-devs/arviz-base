# File generated with docstub

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import lazy_loader as _lazy
import numpy as np
import numpyro
from _typeshed import Incomplete
from numpy.typing import ArrayLike
from xarray import DataTree

from arviz_base.base import dict_to_dataset, requires
from arviz_base.rcparams import rc_context, rcParams
from arviz_base.utils import expand_dims

if TYPE_CHECKING:
    import jax
    import numpyro
else:
    jax: Incomplete
    numpyro: Incomplete

class NumPyroInferenceAdapter(ABC):
    def __init__(
        self,
        inference_obj: Any,
        model: Callable,
        model_args: tuple,
        model_kwargs: dict,
        sample_shape: tuple[int],
    ) -> None: ...
    @property
    @abstractmethod
    def sample_dims(self) -> list[str]: ...
    @abstractmethod
    def get_samples(
        self, seed: int | None = ..., **kwargs: dict
    ) -> dict[str, ArrayLike]: ...
    def get_sample_stats(self, **kwargs) -> dict[str, ArrayLike]: ...

class SVIAdapter(NumPyroInferenceAdapter):
    def __init__(
        self,
        svi: numpyro.infer.SVI,
        *,
        svi_result: numpyro.infer.svi.SVIRunResult,
        model_args: tuple | None = ...,
        model_kwargs: dict | None = ...,
        num_samples: int = ...,
    ) -> None: ...
    @property
    def sample_dims(self) -> list[str]: ...
    def get_samples(
        self, seed: int | None = ..., **kwargs: dict
    ) -> dict[str, ArrayLike]: ...

class MCMCAdapter(NumPyroInferenceAdapter):
    def __init__(self, mcmc: numpyro.infer.MCMC) -> None: ...
    @property
    def sample_dims(self) -> list[str]: ...
    def get_samples(
        self, seed: int | None = ..., **kwargs: dict
    ) -> dict[str, ArrayLike]: ...
    def get_sample_stats(self, **kwargs: Incomplete) -> dict[str, ArrayLike]: ...

def _add_dims(
    dims_a: dict[str, list[str]], dims_b: dict[str, list[str]]
) -> dict[str, list[str]]: ...
def infer_dims(
    model: Callable,
    model_args: tuple[Any, ...] | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
) -> dict[str, list[str]]: ...

class NumPyroConverter:

    model: Incomplete

    def __init__(
        self,
        *,
        posterior: NumPyroInferenceAdapter | None = ...,
        prior: dict | None = ...,
        posterior_predictive: dict | None = ...,
        predictions: dict | None = ...,
        constant_data: dict | None = ...,
        predictions_constant_data: dict | None = ...,
        log_likelihood: bool = ...,
        index_origin: int | None = ...,
        coords: dict | None = ...,
        dims: dict[str, list[str]] | None = ...,
        pred_dims: dict | None = ...,
        extra_event_dims: dict | None = ...,
        num_chains: int | None = ...,
    ) -> None: ...
    def _get_model_trace(self, model, model_args, model_kwargs, key) -> None: ...
    def _infer_sample_shape(self) -> None: ...
    def posterior_to_xarray(self) -> None: ...
    def sample_stats_to_xarray(self) -> None: ...
    def log_likelihood_to_xarray(self) -> None: ...
    def translate_posterior_predictive_dict_to_xarray(self, dct, dims) -> None: ...
    def posterior_predictive_to_xarray(self) -> None: ...
    def predictions_to_xarray(self) -> None: ...
    def priors_to_xarray(self) -> None: ...
    def observed_data_to_xarray(self) -> None: ...
    def constant_data_to_xarray(self) -> None: ...
    def predictions_constant_data_to_xarray(self) -> None: ...
    def to_datatree(self) -> None: ...
    def infer_dims(self) -> dict[str, list[str]]: ...
    def infer_pred_dims(self) -> dict[str, list[str]]: ...

def from_numpyro(
    posterior: numpyro.infer.MCMC | NumPyroInferenceAdapter | None = ...,
    *,
    prior: dict | None = ...,
    posterior_predictive: dict | None = ...,
    predictions: dict | None = ...,
    constant_data: dict | None = ...,
    predictions_constant_data: dict | None = ...,
    log_likelihood: bool = ...,
    index_origin: int | None = ...,
    coords: dict | None = ...,
    dims: dict[str, list[str]] | None = ...,
    pred_dims: dict | None = ...,
    extra_event_dims: dict | None = ...,
    sample_dims: list[str] | None = ...,
    num_chains: int | None = ...,
) -> DataTree: ...
def from_numpyro_svi(
    svi: numpyro.infer.SVI | None = ...,
    *,
    svi_result: numpyro.infer.svi.SVIRunResult | None = ...,
    model_args: tuple | None = ...,
    model_kwargs: dict | None = ...,
    prior: dict | None = ...,
    posterior_predictive: dict | None = ...,
    predictions: dict | None = ...,
    constant_data: dict | None = ...,
    predictions_constant_data: dict | None = ...,
    log_likelihood: bool = ...,
    index_origin: int | None = ...,
    coords: dict | None = ...,
    dims: dict[str, list[str]] | None = ...,
    pred_dims: dict | None = ...,
    extra_event_dims: dict | None = ...,
    num_samples: int = ...,
) -> DataTree: ...
