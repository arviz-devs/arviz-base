# File generated with docstub

import re
import warnings
from collections.abc import Hashable, Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from xarray import DataArray, Dataset

def _check_tilde_start(x: Any) -> bool: ...
def _var_names(
    var_names: Hashable | Sequence[Hashable] | None,
    data: Dataset | Sequence[Dataset],
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    check_if_present: bool = ...,
) -> list[Hashable] | None: ...
def _subset_list(
    subset: Hashable | Sequence[Hashable] | None,
    whole_list: Sequence[Hashable],
    filter_items: Literal[None, "like", "regex"] | None = ...,
    warn: bool = ...,
    check_if_present: bool = ...,
) -> list[Hashable] | None: ...
def _get_coords(
    data: DataArray | Dataset | Sequence[DataArray | Dataset],
    coords: dict[Any, ArrayLike] | Sequence[dict[Any, ArrayLike]],
) -> DataArray | Dataset | list[DataArray | Dataset]: ...
def expand_dims(x: ArrayLike) -> NDArray: ...
