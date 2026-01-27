# File generated with docstub

from collections.abc import Hashable, Iterable, Sequence
from numbers import Number
from typing import Literal

import numpy as np
import pandas
import pandas as pd
import xarray as xr
from _typeshed import Incomplete
from numpy.typing import ArrayLike
from xarray import DataArray, Dataset

from arviz_base.converters import convert_to_dataset
from arviz_base.labels import BaseLabeller
from arviz_base.rcparams import rcParams
from arviz_base.sel_utils import xarray_sel_iter
from arviz_base.utils import _var_names

from .labels import Labeller

__all__ = [
    "dataset_to_dataarray",
    "dataset_to_dataframe",
    "explode_dataset_dims",
    "extract",
    "references_to_dataset",
]

def extract(
    data: Incomplete,
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
) -> DataArray | Dataset: ...
def _stratified_resample(weights: Incomplete, rng: Incomplete) -> None: ...
def dataset_to_dataarray(
    ds: Dataset,
    sample_dims: Sequence[Hashable] | None = ...,
    labeller: Labeller | None = ...,
    add_coords: bool = ...,
    new_dim: Hashable = ...,
    label_type: Literal["flat", "vert"] = ...,
) -> DataArray: ...
def dataset_to_dataframe(
    ds: Dataset,
    sample_dims: Sequence[Hashable] | None = ...,
    labeller: Labeller | None = ...,
    multiindex: Literal["row", "column"] | bool = ...,
    new_dim: Hashable = ...,
) -> pandas.DataFrame: ...
def explode_dataset_dims(
    ds: Dataset, dim: Hashable | Sequence[Hashable], labeller: Labeller | None = ...
) -> Dataset: ...
def references_to_dataset(
    references: Incomplete,
    ds: Dataset,
    sample_dims: Iterable[Hashable] | None = ...,
    ref_dim: str | list | None = ...,
) -> Dataset: ...
