# File generated with docstub

import warnings
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

from numpy.typing import ArrayLike
from xarray import DataTree

from arviz_base.base import dict_to_dataset
from arviz_base.rcparams import rcParams

def from_dict(
    data: Mapping[Any, Mapping[Any, ArrayLike]],
    *,
    name: str | None = ...,
    sample_dims: Sequence[Hashable] | None = ...,
    save_warmup: bool | None = ...,
    index_origin: int | None = ...,
    coords: Mapping[Any, ArrayLike] | None = ...,
    dims: Mapping[Any, Sequence[Hashable]] | None = ...,
    pred_dims: Mapping[Any, Sequence[Hashable]] | None = ...,
    pred_coords: Mapping[Any, ArrayLike] | None = ...,
    check_conventions: bool = ...,
    attrs: Mapping[Any, Mapping[Any, Any]] | None = ...,
) -> DataTree: ...
