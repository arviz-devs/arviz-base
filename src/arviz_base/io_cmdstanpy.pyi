# File generated with docstub

import logging
import re
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cmdstanpy
import numpy as np
from _typeshed import Incomplete
from numpy.typing import ArrayLike
from xarray import DataTree

from arviz_base.base import dict_to_dataset, infer_stan_dtypes, requires
from arviz_base.rcparams import rcParams

_log: Incomplete

class CmdStanPyConverter:
    def __init__(
        self,
        *,
        posterior=...,
        posterior_predictive=...,
        predictions=...,
        prior=...,
        prior_predictive=...,
        observed_data=...,
        constant_data=...,
        predictions_constant_data=...,
        log_likelihood=...,
        index_origin=...,
        coords=...,
        dims=...,
        save_warmup=...,
        dtypes=...,
    ) -> None: ...
    def _warmup_return_to_dict(
        self, data: Incomplete, data_warmup: Incomplete, group: Incomplete
    ) -> None: ...
    def posterior_to_xarray(self) -> None: ...
    def sample_stats_to_xarray(self) -> None: ...
    def sample_stats_prior_to_xarray(self) -> None: ...
    def stats_to_xarray(self, fit) -> None: ...
    def posterior_predictive_to_xarray(self) -> None: ...
    def prior_predictive_to_xarray(self) -> None: ...
    def predictive_to_xarray(self, names, fit) -> None: ...
    def predictions_to_xarray(self) -> None: ...
    def log_likelihood_to_xarray(self) -> None: ...
    def prior_to_xarray(self) -> None: ...
    def observed_data_to_xarray(self) -> None: ...
    def constant_data_to_xarray(self) -> None: ...
    def predictions_constant_data_to_xarray(self) -> None: ...
    def to_datatree(self) -> None: ...

def _as_set(spec) -> None: ...
def _filter(names, spec) -> None: ...
def _unpack_fit(fit, items: list, save_warmup: bool, dtypes: dict) -> dict: ...
def from_cmdstanpy(
    posterior: cmdstanpy.CmdStanMCMC | None = ...,
    *,
    posterior_predictive: str | list[str] | None = ...,
    predictions: str | list[str] | None = ...,
    prior: cmdstanpy.CmdStanMCMC | None = ...,
    prior_predictive: str | list[str] | None = ...,
    observed_data: Mapping[str, ArrayLike] | None = ...,
    constant_data: Mapping[str, ArrayLike] | None = ...,
    predictions_constant_data: dict | None = ...,
    log_likelihood: str | list[str] | dict[str, str] | None = ...,
    index_origin: int | None = ...,
    coords: dict | None = ...,
    dims: Mapping[Any, Sequence[Hashable]] | None = ...,
    save_warmup: bool | None = ...,
    dtypes: dict | None = ...,
) -> DataTree: ...
