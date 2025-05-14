# File generated with docstub

from collections.abc import Hashable, Mapping, Sequence

import cmdstanpy
import xarray
from _typeshed import Incomplete

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
    posterior_predictive: str | None = ...,
    predictions: str | None = ...,
    prior: cmdstanpy.CmdStanMCMC | None = ...,
    prior_predictive: str | None = ...,
    observed_data: dict | None = ...,
    constant_data: dict | None = ...,
    predictions_constant_data: dict | None = ...,
    log_likelihood: str | None = ...,
    index_origin: int | None = ...,
    coords: dict | None = ...,
    dims: Mapping[Hashable, Sequence[Hashable]] | None = ...,
    save_warmup: bool | None = ...,
    dtypes: dict | None = ...,
) -> xarray.DataTree: ...
