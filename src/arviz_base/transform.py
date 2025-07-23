"""Transform helper."""

from collections.abc import Callable
from copy import deepcopy

import xarray as xr

from arviz_base.utils import _var_names


def get_unconstrained_samples(
    idata,
    var_names=None,
    filter_vars=None,
    transform_funcs: dict[str, Callable[[xr.DataArray], xr.DataArray]] | None = None,
    group="posterior",
    return_dataset=False,
):
    """Transform helper.

    It transform the variables of a DataTree in a given group applying the functions
    in the dictionary `transform_funcs`.
    With the default values, it would combine the samples in posterior and unconstrained posterior
    to have all the samples in the unconstrained space.

    Parameters
    ----------
    idata : DataTree-like
        DataTree from which to extract the data.
    var_names : str or list of str, optional
        Variables to be extracted. Prefix the variables by `~` when you want to exclude them.
    filter_vars : {None, "like", "regex"}, optional
        If `None` (default), interpret var_names as the real variables names. If "like",
        interpret var_names as substrings of the real variables names. If "regex",
        interpret var_names as regular expressions on the real variables names. A la
        `pandas.filter`.
    transform_funcs : dict of {str : Callable}, optional
        Dictionary with the functions to transform each variable, by default None.
        Each function *must* accept a single DataArray and return also a
        DataArray.
        If `None`, the values are taking from the group `unconstrained_{group}.
    group : str, optional
        The group where the transformation will be applied, by default `posterior`
    return_dataset : bool, optional
        If true it will return a Dataset with the transformed samples, by default False.

    Returns
    -------
    DataTree-like or Dataset
        By defaults, it returns the original DataTree with the transformed data in
        the `unconstrained_{group}` group.
        A Dataset with the transformed samples.
    """
    if transform_funcs is None:
        transform_funcs = {}

    # get the two datasets and extract variables
    idata = deepcopy(idata)
    ds_in = idata[group].to_dataset()
    var_names_in = _var_names(var_names, ds_in, filter_vars)
    if var_names_in:
        ds_in = ds_in[var_names_in]
    ds_out = idata.get(f"unconstrained_{group}", None)
    if ds_out:
        ds_out = ds_out.to_dataset()

    # for each variable in the input group, decide what to store
    new_vars = {}
    for name, da in ds_in.data_vars.items():
        if name in transform_funcs:
            # validate function
            func = transform_funcs[name]
            if not callable(func):
                raise TypeError(f"transform_funcs[{name!r}] must be callable, got {type(func)}")
            # apply user‐supplied transform to the datarray
            new_da = func(da)
            if not isinstance(new_da, xr.DataArray):
                raise TypeError(
                    f"transform_funcs[{name!r}] must return an xarray.DataArray,"
                    f" got {type(new_da)} instead."
                )
        elif ds_out is not None and name in ds_out.data_vars:
            # reuse the existing unconstrained version
            new_da = ds_out[name]
        else:
            # no transform, no unconstrained: copy the original
            new_da = da

        new_vars[name] = new_da

    # rebuild a Dataset for the return_group (keep the same coords)
    ds_new = xr.Dataset(new_vars, attrs=ds_in.attrs)

    if return_dataset:
        # return dataset
        return ds_new
    else:
        # return datatree in the unconstrained_{group}
        idata[f"unconstrained_{group}"] = xr.DataTree(ds_new)
        return idata
