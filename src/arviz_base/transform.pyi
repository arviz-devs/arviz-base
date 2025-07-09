# File generated with docstub

from collections.abc import Callable

import xarray
import xarray as xr

def transform(
    idata: xarray.DataTree,
    transform_funcs: dict[str, Callable] | None = ...,
    group: str = ...,
    return_group: str = ...,
) -> xarray.DataTree: ...
