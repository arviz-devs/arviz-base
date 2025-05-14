# File generated with docstub

import types
from collections.abc import Callable, Hashable, Iterable
from typing import TYPE_CHECKING

import xarray
from _typeshed import Incomplete
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    pass

RequiresArgTypeT: Incomplete
RequiresReturnTypeT: Incomplete

def generate_dims_coords(
    shape: Iterable[int],
    var_name: Iterable[Hashable],
    dims: Iterable[Hashable] | None = ...,
    coords: dict[Hashable, ArrayLike] | None = ...,
    index_origin: int | None = ...,
    skip_event_dims: bool = ...,
    check_conventions: bool = ...,
) -> tuple[list[Hashable], dict[Hashable, NDArray]]: ...
def ndarray_to_dataarray(
    ary: ArrayLike,
    var_name: Hashable,
    *,
    dims: Iterable[Hashable] | None = ...,
    sample_dims: Iterable[Hashable] | None = ...,
    coords: dict[Hashable, ArrayLike] | None = ...,
    index_origin: int | None = ...,
    skip_event_dims: bool = ...,
    check_conventions: bool = ...,
) -> xarray.DataArray: ...
def dict_to_dataset(
    data: dict[Hashable, ArrayLike],
    *,
    attrs: dict | None = ...,
    inference_library: types.ModuleType | None = ...,
    coords: dict[Hashable, ArrayLike] | None = ...,
    dims: dict[Hashable, Iterable[Hashable]] | None = ...,
    sample_dims: Iterable[Hashable] | None = ...,
    index_origin: int | None = ...,
    skip_event_dims: bool = ...,
    check_conventions: bool = ...,
) -> xarray.Dataset: ...
def make_attrs(
    attrs: dict | None = ..., inference_library: types.ModuleType | None = ...
) -> dict: ...

class requires:
    def __init__(self, *props: str | list[str]) -> None: ...
    def __call__(
        self, func: Callable[[RequiresArgTypeT], RequiresReturnTypeT]
    ) -> Callable[[RequiresArgTypeT], RequiresReturnTypeT | None]: ...

def infer_stan_dtypes(stan_code) -> None: ...
