# File generated with docstub

import logging
import re
from collections.abc import Iterable
from math import ceil

import numpy as np
import stan
from _typeshed import Incomplete
from xarray import DataArray, Dataset, DataTree

from arviz_base.base import (
    dict_to_dataset,
    generate_dims_coords,
    infer_stan_dtypes,
    make_attrs,
    requires,
)
from arviz_base.rcparams import rcParams

_log: Incomplete

class PyStanConverter:
    def __init__(
        self,
        *,
        posterior: Incomplete = ...,
        posterior_model: Incomplete = ...,
        posterior_predictive: Incomplete = ...,
        predictions: Incomplete = ...,
        prior: Incomplete = ...,
        prior_model: Incomplete = ...,
        prior_predictive: Incomplete = ...,
        observed_data: Incomplete = ...,
        constant_data: Incomplete = ...,
        predictions_constant_data: Incomplete = ...,
        log_likelihood: Incomplete = ...,
        coords: Incomplete = ...,
        dims: Incomplete = ...,
        save_warmup: Incomplete = ...,
        dtypes: Incomplete = ...,
    ) -> None: ...
    def _warmup_return_to_dict(
        self,
        data: Incomplete,
        data_warmup: Incomplete,
        group: Incomplete,
        attrs: Incomplete = ...,
    ) -> None: ...
    def posterior_to_xarray(self) -> None: ...
    def sample_stats_to_xarray(self) -> None: ...
    def log_likelihood_to_xarray(self) -> None: ...
    def posterior_predictive_to_xarray(self) -> None: ...
    def predictions_to_xarray(self) -> None: ...
    def prior_to_xarray(self) -> None: ...
    def sample_stats_prior_to_xarray(self) -> None: ...
    def prior_predictive_to_xarray(self) -> None: ...
    def observed_data_to_xarray(self) -> None: ...
    def constant_data_to_xarray(self) -> None: ...
    def predictions_constant_data_to_xarray(self) -> None: ...
    def to_datatree(self) -> None: ...

def get_draws(
    fit: Incomplete,
    model: Incomplete = ...,
    variables: Incomplete = ...,
    ignore: Incomplete = ...,
    warmup: Incomplete = ...,
    dtypes: Incomplete = ...,
) -> None: ...
def get_sample_stats(
    fit: Incomplete,
    variables: Incomplete = ...,
    ignore: Incomplete = ...,
    warmup: Incomplete = ...,
    dtypes: Incomplete = ...,
) -> None: ...
def get_attrs(fit: Incomplete, model: Incomplete = ...) -> None: ...
def infer_dtypes(fit: Incomplete, model: Incomplete = ...) -> None: ...
def from_pystan(
    posterior: stan.fit.Fit | None = ...,
    *,
    posterior_predictive: str | None = ...,
    predictions: str | None = ...,
    prior: stan.fit.Fit | None = ...,
    prior_predictive: str | None = ...,
    observed_data: str | None = ...,
    constant_data: str | None = ...,
    predictions_constant_data: str | None = ...,
    log_likelihood: dict[str, str] | None = ...,
    coords: dict[str, Iterable] | None = ...,
    dims: dict[str, list[str]] | None = ...,
    posterior_model: stan.model.Model | None = ...,
    prior_model: stan.model.Model | None = ...,
    save_warmup: bool | None = ...,
    dtypes: dict | None = ...,
) -> DataTree: ...
