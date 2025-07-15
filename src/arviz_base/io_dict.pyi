# File generated with docstub

import warnings
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

import xarray
from numpy.typing import ArrayLike
from xarray import DataTree

from arviz_base.base import dict_to_dataset
from arviz_base.rcparams import rcParams

def from_dict(
    data: Mapping[Hashable, Mapping[Hashable, ArrayLike]],
    *,
    name: str | None = ...,
    sample_dims: Iterable[Hashable] | None = ...,
    save_warmup: bool | None = ...,
    index_origin: int | None = ...,
    coords: Mapping[Hashable, ArrayLike] | None = ...,
    dims: Mapping[Hashable, Sequence[Hashable]] | None = ...,
    pred_dims: Mapping[Hashable, Sequence[Hashable]] | None = ...,
    pred_coords: dict[str, list] | None = ...,
    check_conventions: bool = ...,
    attrs: Mapping[Hashable, Mapping[Hashable, Any]] | None = ...,
) -> xarray.DataTree: ...
