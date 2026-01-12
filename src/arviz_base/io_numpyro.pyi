# File generated with docstub

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import numpyro
from _typeshed import Incomplete
from xarray import Dataset, DataTree

from arviz_base.base import dict_to_dataset, requires
from arviz_base.rcparams import rc_context, rcParams
from arviz_base.utils import expand_dims

def _add_dims(
    dims_a: dict[str, list[str]], dims_b: dict[str, list[str]]
) -> dict[str, list[str]]: ...
def infer_dims(
    model: Callable,
    model_args: tuple[Any, ...] | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
) -> dict[str, list[str]]: ...

class BaseNumPyroConverter(ABC):

    model: Incomplete
    nchains: Incomplete
    ndraws: Incomplete

    def __init__(
        self,
        *,
        posterior: (
            numpyro.infer.mcmc.MCMC | numpyro.infer.svi.SVI | object | None
        ) = ...,
        prior: dict | None = ...,
        posterior_predictive: dict | None = ...,
        predictions: dict | None = ...,
        constant_data: dict | None = ...,
        predictions_constant_data: dict | None = ...,
        log_likelihood=...,
        index_origin: int | None = ...,
        coords: dict | None = ...,
        dims: dict[str, list[str]] | None = ...,
        pred_dims: dict | None = ...,
        extra_event_dims: dict | None = ...,
        num_chains: int = ...,
    ) -> None: ...
    @abstractmethod
    def _extract_samples_from_posterior(self, posterior) -> None: ...
    @abstractmethod
    def _prepare_predictive_data(self, dct: dict) -> dict: ...
    @abstractmethod
    def sample_stats_to_xarray(self) -> Dataset | None: ...
    def _get_model_trace(self, model, model_args, model_kwargs, key) -> None: ...
    def posterior_to_xarray(self) -> None: ...
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

class MCMCConverter(BaseNumPyroConverter):
    def _extract_samples_from_posterior(self, posterior) -> None: ...
    def _prepare_predictive_data(self, dct: dict) -> dict: ...
    def sample_stats_to_xarray(self) -> None: ...

class SVIConverter(BaseNumPyroConverter):
    def __init__(
        self,
        svi,
        *,
        svi_result,
        model_args=...,
        model_kwargs=...,
        num_samples=...,
        **base_kwargs,
    ) -> None: ...
    def _create_posterior_wrapper(self) -> None: ...
    def _extract_samples_from_posterior(self, posterior) -> None: ...
    def _prepare_predictive_data(self, dct: dict) -> dict: ...
    def sample_stats_to_xarray(self) -> None: ...

def from_numpyro(
    posterior: numpyro.infer.mcmc.MCMC | None = ...,
    *,
    prior: dict | None = ...,
    posterior_predictive: dict | None = ...,
    predictions: dict | None = ...,
    constant_data: dict | None = ...,
    predictions_constant_data: dict | None = ...,
    log_likelihood=...,
    index_origin: int | None = ...,
    coords: dict | None = ...,
    dims: dict[str, list[str]] | None = ...,
    pred_dims: dict | None = ...,
    extra_event_dims: dict | None = ...,
    num_chains: int = ...,
) -> DataTree: ...
def from_numpyro_svi(
    svi: numpyro.infer.svi.SVI,
    *,
    svi_result: numpyro.infer.svi.SVIRunResult,
    model_args: tuple | None = ...,
    model_kwargs: dict | None = ...,
    prior: dict | None = ...,
    posterior_predictive: dict | None = ...,
    predictions: dict | None = ...,
    constant_data: dict | None = ...,
    predictions_constant_data: dict | None = ...,
    log_likelihood=...,
    index_origin: int | None = ...,
    coords: dict | None = ...,
    dims: dict[str, list[str]] | None = ...,
    pred_dims: dict | None = ...,
    extra_event_dims: dict | None = ...,
    model=...,
    num_samples: int = ...,
) -> DataTree: ...
