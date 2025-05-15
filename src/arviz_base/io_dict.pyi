# File generated with docstub

import warnings
from collections.abc import Hashable, Iterable

import xarray
from numpy.typing import ArrayLike
from xarray import DataTree

from arviz_base.base import dict_to_dataset
from arviz_base.rcparams import rcParams

def from_dict(
    data: dict[str, dict[Hashable, ArrayLike]],
    *,
    name: str | None = ...,
    sample_dims: Iterable[Hashable] | None = ...,
    save_warmup: bool | None = ...,
    index_origin: int | None = ...,
    coords: dict[Hashable, list] | None = ...,
    dims: dict[Hashable, list[Hashable]] | None = ...,
    pred_dims: dict[str, list[str]] | None = ...,
    pred_coords: dict[str, list] | None = ...,
    check_conventions: bool = ...,
    attrs: dict[str, dict] | None = ...,
) -> xarray.DataTree: ...
