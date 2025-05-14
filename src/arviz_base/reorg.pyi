# File generated with docstub

from collections.abc import Hashable, Iterable, Sequence
from typing import Literal

import numpy as np
import pandas
import xarray
from numpy.typing import ArrayLike

from .labels import Labeller

__all__ = [
    "dataset_to_dataarray",
    "dataset_to_dataframe",
    "explode_dataset_dims",
    "extract",
    "references_to_dataset",
]

def extract(
    data,
    group: str = ...,
    sample_dims: Sequence[Hashable] | None = ...,
    *,
    combined: bool = ...,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    num_samples: int | None = ...,
    weights: ArrayLike | None = ...,
    resampling_method: str | None = ...,
    keep_dataset: bool = ...,
    random_seed: int | None = ...,
) -> xarray.DataArray | xarray.Dataset: ...
def _stratified_resample(weights, rng) -> None: ...
def dataset_to_dataarray(
    ds: xarray.Dataset,
    sample_dims: Sequence[Hashable] | None = ...,
    labeller: Labeller | None = ...,
) -> xarray.DataArray: ...
def dataset_to_dataframe(
    ds: xarray.Dataset,
    sample_dims: Sequence[Hashable] | None = ...,
    labeller: Labeller | None = ...,
    multiindex: bool = ...,
) -> pandas.DataFrame: ...
def explode_dataset_dims(
    ds: xarray.Dataset,
    dim: Hashable | Sequence[Hashable],
    labeller: Labeller | None = ...,
) -> xarray.Dataset: ...
def references_to_dataset(
    references: np.ScalarType,
    ds: xarray.Dataset,
    sample_dims: Iterable[Hashable] | None = ...,
    ref_dim: str | list | None = ...,
) -> xarray.Dataset: ...
