# File generated with docstub

import re
import warnings
from collections.abc import Hashable, Sequence
from typing import Literal

import numpy as np
import xarray
from _typeshed import Incomplete
from numpy.typing import ArrayLike

def _check_tilde_start(x: Incomplete) -> None: ...
def _var_names(
    var_names: str | list | None,
    data: xarray.Dataset | Sequence[xarray.Dataset],
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    check_if_present: bool = ...,
) -> list | None: ...
def _subset_list(
    subset: str,
    whole_list: list,
    filter_items: Literal[None, "like", "regex"] | None = ...,
    warn=...,
    check_if_present=...,
) -> list | None: ...
def _get_coords(
    data: xarray.DataArray, coords: dict[Hashable, ArrayLike]
) -> xarray.Dataset | xarray.DataArray: ...
def expand_dims(x) -> None: ...
