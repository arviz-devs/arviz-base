"""Transform helper."""

import xarray as xr


def transform(
    idata,
    transform_funcs=None,
    group="posterior",
    return_group="unconstrained_posterior",
):
    """Transform helper.

    It transform the variables of a DataTree in a given group applying the functions
    in the dictionary ``transform_funcs``.
    With the default values, it would combine the samples in posterior and unconstrained posterior
    to have all the samples in the unconstrained space.

    Parameters
    ----------
    idata : DataTree-like
        DataTree from which to extract the data.
    transform_funcs : dict of {str : Callable}, optional
        Dictionary with the functions to transform each variable, by default None
        If ``None``, the values are taking from the group ``unconstrained_posterior`` .
    group : str, optional
        The group where the transformation will be applied, by default "posterior"
    return_group : str, optional
        The group where the transformed data will be saved, by default "unconstrained_posterior"

    Returns
    -------
    DataTree-like
        DataTree with the transformed data in the `return_group`.
    """
    if transform_funcs is None:
        transform_funcs = {}

    # get the two datasets
    ds_in = idata[group]
    ds_out = idata.get(return_group, None)

    # for each variable in the input group, decide what to store
    new_vars = {}
    for name, da in ds_in.data_vars.items():
        if name in transform_funcs:
            # apply user‐supplied transform to the numpy values
            arr = da.values
            new_arr = transform_funcs[name](arr)
            new_da = xr.DataArray(
                new_arr,
                dims=da.dims,
                coords=da.coords,
                attrs=da.attrs,
            )
        elif ds_out is not None and name in ds_out.data_vars:
            # reuse the existing unconstrained version
            new_da = ds_out[name]
        else:
            # no transform, no unconstrained: copy the original
            new_da = da

        new_vars[name] = new_da

    # rebuild a Dataset for the return_group (keep the same coords)
    ds_new = xr.Dataset(new_vars, coords=ds_in.coords, attrs=ds_in.attrs)

    return xr.DataTree.from_dict({return_group: ds_new})
