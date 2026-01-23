# File generated with docstub

import re
import warnings
from collections.abc import Hashable, Sequence
from typing import Literal

import numpy as np
from _typeshed import Incomplete
from numpy.typing import ArrayLike
from xarray import DataArray, Dataset

def _check_tilde_start(x: Incomplete) -> None: ...
def _var_names(
    var_names: str | list | None,
    data: Dataset | Sequence[Dataset],
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    check_if_present: bool = ...,
) -> list | None: ...
def _subset_list(
    subset: str,
    whole_list: list,
    filter_items: Literal[None, "like", "regex"] | None = ...,
    warn: Incomplete = ...,
    check_if_present: Incomplete = ...,
) -> list | None: ...
def _get_coords(
    data: DataArray, coords: dict[Hashable, ArrayLike]
) -> Dataset | DataArray: ...
def expand_dims(x: Incomplete) -> None: ...
