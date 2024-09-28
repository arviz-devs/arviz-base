"""Utilities for selecting and iterating on xarray objects."""

from itertools import product, tee

import numpy as np
import xarray as xr

from arviz_base.labels import BaseLabeller
from arviz_base.rcparams import rcParams

__all__ = ["xarray_sel_iter", "xarray_var_iter", "xarray_to_ndarray"]


def _dims(data, var_name, skip_dims):
    return [dim for dim in data[var_name].dims if dim not in skip_dims]


def _zip_dims(new_dims, vals):
    return [dict(zip(new_dims, prod)) for prod in product(*vals)]


def xarray_sel_iter(
    data, var_names=None, combined=None, skip_dims=None, dim_to_idx=None, reverse_selections=False
):
    """Convert xarray data to an iterator over variable names and selections.

    Iterates over each var_name and all of its dimensions, returning the variable
    names and selections that allow properly obtain the data subsets from ``data`` as desired.
    The iterable returned defines an exhaustive collection of subsets. Both the
    input object and the selections defined can have any dimensionality, and
    the selections from each element in the iterable can have different dimensionality
    between them.

    When looping within a dimension, this can be done over the dimension itself or
    via unique items of explicit indexes for that dimension.

    Parameters
    ----------
    data : Dataset or DataArray
        Posterior data in an xarray
    var_names : iterator of hashable, optional
        Should be a subset of data.data_vars. Defaults to all of them.
    combined : bool, optional
        Whether to combine chains or leave them separate. By default (``None``),
        this is ignored and the `chain` dimension is looped over or skipped
        based on `skip_dims`. If set to ``True``/``False`` then `skip_dims`
        is modified in order to ensure combining chains or not.
    skip_dims : set, optional
        Dimensions to not iterate over. Defaults to rcParam ``data.sample_dims``.
    dim_to_idx : mapping {hashable : hashable}, optional
        Mapping from dimensions to indexes to loop over these dimensions using unique
        items in the provided index.
    reverse_selections : bool
        Whether to reverse selections before iterating.

    Yields
    ------
    var_name : str
        Variable name to which `selection` and `iselection` correspond to.
    selection : dict of {hashable : any}
        Keys are coordinate names and values are scalar coordinate values.
        To get the values of the variable at these coordinates, do
        ``data[var_name].sel(selection)`` for :class:`~xarray.Dataset` or
        ``data.sel(selection)`` for :class:`~xarray.DataArray`.
    iselection : dict of {hashable : any}
        Keys are dimension names and values are positional indexes (might not be scalars).
        To get the values of the variable at these coordinates, do
        ``data[var_name].isel(iselection)`` for :class:`~xarray.Dataset` or
        ``data.isel(iselection)`` for :class:`~xarray.DataArray`.

    Examples
    --------
    Let's create a 3d :class:`~xarray.DataArray` with dimensions "chain", "draw"
    and "obs_dim".

    .. jupyter-execute::

        import xarray as xr
        import numpy as np
        from arviz_base import xarray_sel_iter
        xr.set_options(display_expand_data=False, display_expand_indexes=True)

        data = xr.DataArray(
            np.random.default_rng(2).normal(size=(2,3,7)),
            dims=["chain", "draw", "obs_dim"],
            coords={"chain": [1, 2]},
            name="sample"
        )
        data

    By default, ``xarray_sel_iter`` will return an iterable with the subsets that
    are generated from looping over all dimensions not in ``rcParams["data.sample_dims"]``
    (the default value for `skip_dims`). Here, it will be an iterable of length 7,
    selecting each position in the "obs_dim" dimension:

    .. jupyter-execute::

        list(xarray_sel_iter(data))

    Here we are using a ``DataArray``, so the first position in each tuple,
    ``var_name`` is always the same and corresponds to its name.

    If we want to iterate over each _sample_ (pair of "chain", "draw" values)
    we can use `skip_dims`:

    .. jupyter-execute::

        list(xarray_sel_iter(data, skip_dims={"obs_dim"}))

    Now there are 6 elements, 3 values for "draw" times 2 values for "chain". Note also
    how the two returned selections now differ. The _coordinate_ values for "chain"
    are ``1, 2`` whereas their corresponding _positions_ are `0, 1`.

    To go further in the examples, and show the usage of `dim_to_idx` we need to
    add some explicit indexes to the ``DataArray``. We do that by adding new coordinates
    and then setting them as indexes.

    .. jupyter-execute::

        data = data.assign_coords(
            {"obs_id": ("obs_dim", np.arange(7)), "label_id": ("obs_dim", list("babacbc"))}
        ).set_xindex("obs_id").set_xindex("label_id")
        data

    Note that both the "Coordinates" and the "Indexes" sections of the output have been updated.
    We can now loop over the "obs_dim" dimension by "itself" (like we did in the first example)
    or using either of these two new indexes. If we use "label_id", the returned iterator
    will have length 3, as there are only 3 unique values in "label_id", ``a, b, c``.

    .. jupyter-execute::

        list(xarray_sel_iter(data, dim_to_idx={"obs_dim": "label_id"}))

    Note that the order of the coordinate values is preserved. Moreover, now not only
    the values in the selection dict values are different, also their keys. "label_id"
    is a coordinate+index, but not a dimension, so it can not be used for positional
    indexing.

    See Also
    --------
    xarray_var_iter
        Return a similar iterator whose elements also include the selected subset as a DataArray.
    """
    if skip_dims is None:
        skip_dims = set(rcParams["data.sample_dims"])

    if dim_to_idx is None:
        dim_to_idx = {}

    if combined is not None:
        if combined:
            skip_dims.add("chain")
        else:
            skip_dims.remove("chain")

    if var_names is None:
        if isinstance(data, xr.Dataset):
            var_names = list(data.data_vars)
        elif isinstance(data, xr.DataArray):
            var_names = [data.name]
            data = {data.name: data}

    for var_name in var_names:
        if var_name in data:
            new_dims = _dims(data, var_name, skip_dims)
            new_dimsidx = [dim_to_idx.get(dim) if dim in dim_to_idx else dim for dim in new_dims]
            vals = [
                (
                    data[var_name].xindexes[dim_to_idx.get(dim)].to_pandas_index().unique().values
                    if dim in dim_to_idx
                    else data[var_name][dim].values
                )
                for dim in new_dims
            ]
            dims = _zip_dims(new_dimsidx, vals)
            ivals = []
            for i, dim in enumerate(new_dims):
                if dim in dim_to_idx:
                    idx_values = data[var_name][dim_to_idx.get(dim)].values
                    dim_ivals = [np.argwhere(idx_values == v).squeeze() for v in vals[i]]
                    ivals.append([v.item() if v.size == 1 else v for v in dim_ivals])
                else:
                    ivals.append(range(len(vals[i])))
            idims = _zip_dims(new_dims, ivals)
            if reverse_selections:
                dims = reversed(dims)
                idims = reversed(idims)

            for selection, iselection in zip(dims, idims):
                yield var_name, selection, iselection


def xarray_var_iter(
    data,
    var_names=None,
    combined=None,
    skip_dims=None,
    dim_to_idx=None,
    reverse_selections=False,
    dim_order=None,
):
    """Convert xarray data to an iterator over vectors.

    Iterates over each var_name and all of its coordinates, returning selected subsets as
    DataArray.

    Parameters
    ----------
    data : xarray.Dataset
        Posterior data in an xarray
    var_names : iterator of hashable, optional
        Should be a subset of data.data_vars. Defaults to all of them.
        Passed to :func:`~arviz_base.xarray_sel_iter`.
    combined : bool, optional
        Whether to combine chains or leave them separate.
        Passed to :func:`~arviz_base.xarray_sel_iter`.
    skip_dims : set, optional
        Dimensions to not iterate over.
        Passed to :func:`~arviz_base.xarray_sel_iter`.
    dim_to_idx : dict, optional
        Mapping from dimension names to index names to define a different way to
        loop over that dimension.
        Passed to :func:`~arviz_base.xarray_sel_iter`.
    reverse_selections : bool, optional
        Whether to reverse selections before iterating.
        Passed to :func:`~arviz_base.xarray_sel_iter`.
    dim_order: list, optional
        Order for the first dimensions. Skips dimensions not found in the variable.

    Yields
    ------
    var_name : str
        Variable name to which `selection`, `iselection` and `data_subset` correspond to.
    selection : dict of {hashable : any}
        Keys are coordinate names and values are scalar coordinate values.
    iselection : dict of {hashable : any}
        Keys are dimension names and values are positional indexes (might not be scalars).
    data_subset : DataArray)
        Values of the variable at those coordinates.

    See Also
    --------
    xarray_sel_iter
        Return the iterator without the DataArray subset corresponding to the selection.
    """
    data_to_sel = data
    if var_names is None and isinstance(data, xr.DataArray):
        data_to_sel = {data.name: data}

    if isinstance(dim_order, str):
        dim_order = [dim_order]

    for var_name, selection, iselection in xarray_sel_iter(
        data,
        var_names=var_names,
        combined=combined,
        skip_dims=skip_dims,
        dim_to_idx=dim_to_idx,
        reverse_selections=reverse_selections,
    ):
        selected_data = data_to_sel[var_name].sel(selection)
        if dim_order is not None:
            dim_order_selected = [dim for dim in dim_order if dim in selected_data.dims]
            if dim_order_selected:
                selected_data = selected_data.transpose(*dim_order_selected, ...)
        yield var_name, selection, iselection, selected_data.values


def xarray_to_ndarray(data, *, var_names=None, combined=True, label_fun=None):
    """Take xarray data and unpacks into variables and data into list and numpy array respectively.

    Assumes that chain and draw are in coordinates

    Parameters
    ----------
    data: xarray.DataSet
        Data in an xarray from an InferenceData object. Examples include posterior or sample_stats

    var_names: iter
        Should be a subset of data.data_vars not including chain and draws. Defaults to all of them

    combined: bool
        Whether to combine chain into one array

    Returns
    -------
    var_names: list
        List of variable names
    data: np.array
        Data values
    """
    if label_fun is None:
        label_fun = BaseLabeller().make_label_vert
    data_to_sel = data
    if var_names is None and isinstance(data, xr.DataArray):
        data_to_sel = {data.name: data}

    iterator1, iterator2 = tee(xarray_sel_iter(data, var_names=var_names, combined=combined))
    vars_and_sel = list(iterator1)
    unpacked_var_names = [
        label_fun(var_name, selection, isel) for var_name, selection, isel in vars_and_sel
    ]

    # Merge chains and variables, check dtype to be compatible with divergences data
    data0 = data_to_sel[vars_and_sel[0][0]].sel(**vars_and_sel[0][1])
    unpacked_data = np.empty((len(unpacked_var_names), data0.size), dtype=data0.dtype)
    for idx, (var_name, selection, _) in enumerate(iterator2):
        unpacked_data[idx] = data_to_sel[var_name].sel(**selection).values.flatten()

    return unpacked_var_names, unpacked_data
