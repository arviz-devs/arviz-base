# File generated with docstub

from collections.abc import Hashable, Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from xarray import DataArray, Dataset

def _check_tilde_start(x: Any) -> bool: ...
def _var_names(
    var_names: str | list[str] | None,
    data: Dataset | Sequence[Dataset],
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    check_if_present: bool = ...,
) -> list[str] | None: ...
def _subset_list(
    subset: str | list[str] | None,
    whole_list: list[str],
    filter_items: Literal[None, "like", "regex"] | None = ...,
    warn=...,
    check_if_present=...,
) -> list[str] | None: ...
def _get_coords(
    data: DataArray | Dataset | Sequence[DataArray | Dataset],
    coords: dict[Hashable, ArrayLike] | Sequence[dict[Hashable, ArrayLike]],
) -> DataArray | Dataset | list[DataArray | Dataset]: ...
def expand_dims(x: ArrayLike) -> np.ndarray: ...
