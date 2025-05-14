# File generated with docstub

import os

import xarray
from _typeshed import Incomplete

__all__ = ["get_data_home", "clear_data_home", "list_datasets", "load_arviz_data"]

LocalFileMetadata: Incomplete

RemoteFileMetadata: Incomplete
_EXAMPLE_DATA_DIR: Incomplete
_LOCAL_DATA_DIR: Incomplete

with open(os.path.join(_EXAMPLE_DATA_DIR, "data_local.json"), encoding="utf-8") as f:
    LOCAL_DATASETS: Incomplete

with open(os.path.join(_EXAMPLE_DATA_DIR, "data_remote.json"), encoding="utf-8") as f:
    REMOTE_DATASETS: Incomplete

def get_data_home(data_home: str | None = ...) -> None: ...
def clear_data_home(data_home: str | None = ...) -> None: ...
def _sha256(path) -> None: ...
def load_arviz_data(
    dataset: str | None = ..., data_home: str | None = ..., **kwargs: dict
) -> xarray.Dataset: ...
def list_datasets() -> None: ...
