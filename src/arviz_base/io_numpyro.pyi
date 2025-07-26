# File generated with docstub

import warnings

import numpy as np
import numpyro
from _typeshed import Incomplete
from xarray import DataTree

from arviz_base.base import dict_to_dataset, requires
from arviz_base.rcparams import rc_context, rcParams
from arviz_base.utils import expand_dims

class NumPyroConverter:

    model: Incomplete
    nchains: Incomplete
    ndraws: Incomplete

    def __init__(
        self,
        *,
        posterior: numpyro.mcmc.MCMC | None = ...,
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
        num_chains: int = ...,
    ) -> None: ...
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

def from_numpyro(
    posterior: numpyro.mcmc.MCMC | None = ...,
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
    num_chains: int = ...,
) -> DataTree: ...
