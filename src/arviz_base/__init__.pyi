# File generated with docstub

import logging

from _typeshed import Incomplete

_log: Incomplete

from arviz_base import labels, testing
from arviz_base._version import __version__
from arviz_base.base import (
    dict_to_dataset,
    generate_dims_coords,
    make_attrs,
    ndarray_to_dataarray,
)
from arviz_base.citations import citations
from arviz_base.converters import convert_to_dataset, convert_to_datatree
from arviz_base.datasets import (
    clear_data_home,
    get_data_home,
    list_datasets,
    load_arviz_data,
)
from arviz_base.io_cmdstanpy import from_cmdstanpy
from arviz_base.io_dict import from_dict
from arviz_base.io_emcee import from_emcee
from arviz_base.io_numpyro import from_numpyro, from_numpyro_svi
from arviz_base.io_pystan import from_pystan
from arviz_base.rcparams import rc_context, rcParams
from arviz_base.reorg import (
    dataset_to_dataarray,
    dataset_to_dataframe,
    explode_dataset_dims,
    extract,
    references_to_dataset,
)
from arviz_base.sel_utils import xarray_sel_iter, xarray_to_ndarray, xarray_var_iter
from arviz_base.transform import get_unconstrained_samples

__all__ = [
    "__version__",
    "citations",
    "dict_to_dataset",
    "generate_dims_coords",
    "make_attrs",
    "ndarray_to_dataarray",
    "convert_to_dataset",
    "convert_to_datatree",
    "clear_data_home",
    "get_data_home",
    "list_datasets",
    "load_arviz_data",
    "from_cmdstanpy",
    "from_dict",
    "from_emcee",
    "from_numpyro",
    "from_numpyro_svi",
    "labels",
    "rc_context",
    "rcParams",
    "extract",
    "dataset_to_dataarray",
    "dataset_to_dataframe",
    "explode_dataset_dims",
    "references_to_dataset",
    "xarray_sel_iter",
    "xarray_var_iter",
    "xarray_to_ndarray",
    "testing",
    "get_unconstrained_samples",
]
