# File generated with docstub

import emcee
import xarray
from numpy.typing import ArrayLike

def _verify_names(
    sampler: emcee.EnsembleSampler,
    var_names: list[str | None],
    arg_names: list[str | None],
    slices: list | None,
) -> list[str]: ...

class EmceeConverter:
    def __init__(
        self,
        sampler,
        var_names=...,
        slices=...,
        arg_names=...,
        arg_groups=...,
        blob_names=...,
        blob_groups=...,
        index_origin=...,
        coords=...,
        dims=...,
        check_conventions=...,
    ) -> None: ...
    def posterior_to_xarray(self) -> None: ...
    def args_to_xarray(self) -> None: ...
    def blobs_to_dict(self) -> None: ...
    def to_datatree(self) -> None: ...

def from_emcee(
    sampler: emcee.EnsembleSampler | None = ...,
    var_names: list[str] | None = ...,
    slices: list[ArrayLike | slice] | None = ...,
    arg_names: list[str] | None = ...,
    arg_groups: list[str] | None = ...,
    blob_names: list[str] | None = ...,
    blob_groups: list[str] | None = ...,
    index_origin=...,
    coords: dict[str, ArrayLike] | None = ...,
    dims: dict[str, list[str]] | None = ...,
    check_conventions: bool = ...,
) -> xarray.DataTree: ...
