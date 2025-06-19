# File generated with docstub

from collections.abc import Callable, Generator, Hashable, Mapping, Sequence
from itertools import product, tee
from typing import Any

import numpy as np
import xarray
import xarray as xr
from _typeshed import Incomplete
from numpy.typing import NDArray

from arviz_base.labels import BaseLabeller
from arviz_base.rcparams import rcParams

__all__ = ["xarray_sel_iter", "xarray_var_iter", "xarray_to_ndarray"]

def _dims(data: Incomplete, var_name: Incomplete, skip_dims: Incomplete) -> None: ...
def _zip_dims(new_dims: Incomplete, vals: Incomplete) -> None: ...
def xarray_sel_iter(
    data: xarray.Dataset | xarray.DataArray,
    var_names: Sequence[Hashable] | None = ...,
    combined: bool | None = ...,
    skip_dims: set | None = ...,
    dim_to_idx: Mapping[Hashable, Hashable] | None = ...,
    reverse_selections: bool = ...,
) -> Generator[tuple[str, dict[Hashable, Any], dict[Hashable, Any]]]: ...
def xarray_var_iter(
    data: xarray.Dataset,
    var_names: Sequence[Hashable] | None = ...,
    combined: bool | None = ...,
    skip_dims: set | None = ...,
    dim_to_idx: Mapping[Hashable, Hashable] | None = ...,
    reverse_selections: bool = ...,
    dim_order: list | None = ...,
) -> Generator[
    tuple[str, dict[Hashable, Any], dict[Hashable, Any], xarray.DataArray]
]: ...
def xarray_to_ndarray(
    data: xarray.Dataset,
    *,
    var_names: Sequence[Hashable] | None = ...,
    combined: bool = ...,
    label_fun: Callable | None = ...
) -> tuple[list, NDArray]: ...
