# File generated with docstub

from collections.abc import Callable
from copy import deepcopy
from typing import Literal

import xarray
import xarray as xr

from arviz_base.utils import _var_names

def get_unconstrained_samples(
    idata: xarray.DataTree,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    transform_funcs: dict[str, Callable] | None = ...,
    group: str = ...,
    return_dataset: bool = ...,
) -> xarray.DataTree | xarray.Dataset: ...
