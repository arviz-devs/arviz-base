"""Generalistic converters.

Here "generalistic" means catch anything that can be converter into datatree and
convert it via its specific function.
"""

import numpy as np
import xarray as xr
from xarray import DataTree, open_datatree

from arviz_base.base import dict_to_dataset

__all__ = [
    "convert_to_datatree",
    "convert_to_dataset",
]


# pylint: disable=too-many-return-statements
def convert_to_datatree(obj, **kwargs):
    r"""Convert a supported object to a DataTree object following ArviZ conventions.

    This function sends `obj` to the right conversion function. It is idempotent,
    in that it will return DataTree objects unchanged. In general however,
    it is better to call specific conversion functions directly. See below
    for more details.

    Parameters
    ----------
    obj
        A supported object to convert to InferenceData:

         * DataTree: returns unchanged
         * InferenceData: returns the equivalent DataTree. `kwargs` are passed
           to :meth:`datatree.DataTree.from_dict`.
         * str:

           - If it ends with ``.csv``, attempts to load the file as a cmdstan csv fit
             using :func:`from_cmdstan`
           - Otherwise, attempts to load a netcdf or zarr file from disk
             using :func:`open_datatree`

         * pystan fit: Calls :func:`.from_pystan` with default arguments
         * cmdstanpy fit: Calls :func:`from_cmdstanpy` with default arguments
         * cmdstan csv-list: Calls :func:`from_cmdstan` with default arguments
         * emcee sampler: Calls :func:`from_emcee` with default arguments
         * pyro MCMC: Calls :func:`from_pyro` with default arguments
         * numpyro MCMC: calls :func:`from_numpyro` with default arguments
         * beanmachine MonteCarloSamples: Calls :func:`from_beanmachine` with default arguments
         * `xarray.Dataset`: Adds it to the DataTree a the only group. The group name
           is taken from the ``group`` keyword in `kwargs`.
         * `xarray.DataArray`: Adds it to the DataTree as the only variable in a single group.
           If the ``name`` is not set, "x" is used as name. Like above,
           the group name is taken from the ``group`` keyword in `kwargs`.
         * dict: creates an xarray.Dataset with :func:`dict_to_dataset` and adds it
           to the DataTree as the only group (named with the ``group`` key in `kwargs`).
         * `numpy.ndarray`: names the variable "x" and adds it to the DataTree
           with a single group, named with the ``group`` key in `kwargs`.

    kwargs
        Rest of the supported keyword arguments transferred to conversion function.

    Returns
    -------
    DataTree

    See Also
    --------
    from_dict
        Convert a nested dictionary of {group_name: {var_name: data}} to a DataTree.
    """
    kwargs = kwargs.copy()
    group = kwargs.pop("group", "posterior")

    # Cases that convert to DataTree
    if isinstance(obj, DataTree):
        return obj
    if isinstance(obj, str):
        # if obj.endswith(".csv"):
        #     if group == "sample_stats":
        #         kwargs["posterior"] = obj
        #     elif group == "sample_stats_prior":
        #         kwargs["prior"] = obj
        #     return from_cmdstan(**kwargs)
        return open_datatree(obj, **kwargs)
    if obj.__class__.__name__ == "InferenceData":
        return DataTree.from_dict({group: obj[group] for group in obj.groups()}, **kwargs)
    # if (
    #     obj.__class__.__name__ in {"StanFit4Model", "CmdStanMCMC"}
    #     or obj.__class__.__module__ == "stan.fit"
    # ):
    #     if group == "sample_stats":
    #         kwargs["posterior"] = obj
    #     elif group == "sample_stats_prior":
    #         kwargs["prior"] = obj
    #     if obj.__class__.__name__ == "CmdStanMCMC":
    #         return from_cmdstanpy(**kwargs)
    #     return from_pystan(**kwargs)
    # if obj.__class__.__name__ == "EnsembleSampler":  # ugly, but doesn't make emcee a requirement
    #     return from_emcee(sampler=obj, **kwargs)
    # if obj.__class__.__name__ == "MonteCarloSamples":
    #     return from_beanmachine(sampler=obj, **kwargs)
    # if obj.__class__.__name__ == "MCMC" and obj.__class__.__module__.startswith("pyro"):
    #     return from_pyro(posterior=obj, **kwargs)
    # if obj.__class__.__name__ == "MCMC" and obj.__class__.__module__.startswith("numpyro"):
    #     return from_numpyro(posterior=obj, **kwargs)

    # Cases that convert to xarray
    if isinstance(obj, xr.Dataset):
        dataset = obj
    elif isinstance(obj, xr.DataArray):
        if obj.name is None:
            obj.name = "x"
        dataset = obj.to_dataset()
    elif isinstance(obj, dict):
        dataset = dict_to_dataset(obj, **kwargs)
    elif isinstance(obj, np.ndarray):
        dataset = dict_to_dataset({"x": obj}, **kwargs)
    # elif isinstance(obj, (list, tuple)) and isinstance(obj[0], str) and obj[0].endswith(".csv"):
    #     if group == "sample_stats":
    #         kwargs["posterior"] = obj
    #     elif group == "sample_stats_prior":
    #         kwargs["prior"] = obj
    #     return from_cmdstan(**kwargs)
    else:
        allowable_types = (
            "xarray dataarray",
            "xarray dataset",
            "dict",
            "netcdf filename",
            "zarr filename",
            "numpy array",
            # "pystan fit",
            # "emcee fit",
            # "pyro mcmc fit",
            # "numpyro mcmc fit",
            # "cmdstan fit csv filename",
            # "cmdstanpy fit",
            # "beanmachine montecarlosamples",
        )
        raise ValueError(
            f"Can only convert {', '.join(allowable_types)} to InferenceData, "
            f"not {obj.__class__.__name__}"
        )

    return DataTree.from_dict({group: dataset})


def convert_to_dataset(obj, *, group="posterior", **kwargs):
    """Convert a supported object to an xarray.Dataset.

    This function is idempotent: if you pass in an xarray.Dataset, it returns it unchanged.
    If passed another supported object (e.g., dict, DataTree, numpy array), it will be
    converted to a Dataset via `convert_to_datatree`, and the specified group will be extracted.

    Parameters
    ----------
    obj : Any
        A supported object to convert to InferenceData. See `convert_to_datatree` for full list.
    group : str, default "posterior"
        The group name to extract from the converted DataTree.
    **kwargs : dict, optional
        Additional arguments passed to `convert_to_datatree`.

    Returns
    -------
    xarray.Dataset
        The extracted dataset from the converted object.

    Raises
    ------
    ValueError
        If the group cannot be extracted from the resulting DataTree.

    Examples
    --------
    >>> convert_to_dataset({"mu": np.random.randn(500)})
    <xarray.Dataset> ...  # Posterior group with 'mu' variable
    """
    if isinstance(obj, DataTree) and obj.name == group:
        return obj.to_dataset()
    inference_data = convert_to_datatree(obj, group=group, **kwargs)
    dataset = getattr(inference_data, group, None)
    if dataset is None:
        raise ValueError(
            f"Can not extract {group} from {obj}! See docs for other conversion utilities."
        )
    return dataset.to_dataset()
