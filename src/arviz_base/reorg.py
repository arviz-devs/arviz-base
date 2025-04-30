"""Helper functions to reorganize data."""

from numbers import Number

import numpy as np
import pandas as pd
import xarray as xr

from arviz_base.converters import convert_to_dataset
from arviz_base.labels import BaseLabeller
from arviz_base.rcparams import rcParams
from arviz_base.sel_utils import xarray_sel_iter
from arviz_base.utils import _var_names

__all__ = [
    "dataset_to_dataarray",
    "dataset_to_dataframe",
    "explode_dataset_dims",
    "extract",
    "references_to_dataset",
]


# TODO: remove this ignore about too many statements once the code uses validator functions
def extract(  # noqa: PLR0915
    data,
    group="posterior",
    sample_dims=None,
    *,
    combined=True,
    var_names=None,
    filter_vars=None,
    num_samples=None,
    weights=None,
    resampling_method=None,
    keep_dataset=False,
    random_seed=None,
):
    """Extract a group or group subset from a DataTree.

    Parameters
    ----------
    idata : DataTree_like
        DataTree from which to extract the data.
    group : str, optional
        Which group to extract data from.
    sample_dims : sequence of hashable, optional
        List of dimensions that should be considered sampling dimensions.
        Random subsets and potential stacking if ``combine=True`` happen
        over these dimensions only. Defaults to ``rcParams["data.sample_dims"]``.
    combined : bool, optional
        Combine `sample_dims` dimensions into ``sample``. Won't work if
        a dimension named ``sample`` already exists.
        It is irrelevant and ignored when `sample_dims` is a single dimension.
    var_names : str or list of str, optional
        Variables to be extracted. Prefix the variables by `~` when you want to exclude them.
    filter_vars: {None, "like", "regex"}, optional
        If `None` (default), interpret var_names as the real variables names. If "like",
        interpret var_names as substrings of the real variables names. If "regex",
        interpret var_names as regular expressions on the real variables names. A la
        `pandas.filter`.
        Like with plotting, sometimes it's easier to subset saying what to exclude
        instead of what to include
    num_samples : int, optional
        Extract only a subset of the samples. Only valid if ``combined=True`` or
        `sample_dims` represents a single dimension.
    weights : array-like, optional
        Extract a weighted subset of the samples. Only valid if `num_samples` is not ``None``.
    resampling_method : str, optional
        Method to use for resampling. Default is "multinomial". Options are "multinomial"
        and "stratified". For stratified resampling, weights must be provided.
        Default is "stratified" if weights are provided, "multinomial" otherwise.
    keep_dataset : bool, optional
        If true, always return a DataSet. If false (default) return a DataArray
        when there is a single variable.
    random_seed : int, numpy.Generator, optional
        Random number generator or seed. Only used if ``weights`` is not ``None``
        or if ``num_samples`` is not ``None``.

    Returns
    -------
    xarray.DataArray or xarray.Dataset

    Examples
    --------
    The default behaviour is to return the posterior group after stacking the chain and
    draw dimensions.

    .. jupyter-execute::

        import arviz_base as az
        idata = az.load_arviz_data("centered_eight")
        az.extract(idata)

    You can also indicate a subset to be returned, but in variables and in samples:

    .. jupyter-execute::

        az.extract(idata, var_names="theta", num_samples=100)

    To keep the chain and draw dimensions, use ``combined=False``.

    .. jupyter-execute::

        az.extract(idata, group="prior", combined=False)

    """
    # TODO: use validator function
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    if len(sample_dims) == 1:
        combined = True
    if num_samples is not None and not combined:
        raise ValueError(
            "num_samples is only compatible with combined=True or length 1 sample_dims"
        )
    if weights is not None and num_samples is None:
        raise ValueError("weights are only compatible with num_samples")

    data = convert_to_dataset(data, group=group)
    var_names = _var_names(var_names, data, filter_vars)
    if var_names is not None:
        if len(var_names) == 1 and not keep_dataset:
            var_names = var_names[0]
        data = data[var_names]
    elif len(data.data_vars) == 1:
        if keep_dataset:
            data
        else:
            data = data[list(data.data_vars)[0]]

    if weights is not None:
        resampling_method = "stratified" if resampling_method is None else resampling_method
        weights = np.array(weights).ravel()
        if len(weights) != np.prod([data.sizes[dim] for dim in sample_dims]):
            raise ValueError("Weights must have the same size as `sample_dims`")
    else:
        resampling_method = "multinomial" if resampling_method is None else resampling_method

    if resampling_method not in ("multinomial", "stratified"):
        raise ValueError(f"Invalid resampling_method: {resampling_method}")

    if combined and len(sample_dims) != 1:
        data = data.stack(sample=sample_dims)
        combined_dim = "sample"
    elif len(sample_dims) == 1:
        combined_dim = sample_dims[0]

    if weights is not None or num_samples is not None:
        if random_seed is None:
            rng = np.random.default_rng()
        elif isinstance(random_seed, int | np.integer):
            rng = np.random.default_rng(random_seed)
        elif isinstance(random_seed, np.random.Generator):
            rng = random_seed
        else:
            raise ValueError(f"Invalid random_seed value: {random_seed}")

        replace = weights is not None

        if resampling_method == "multinomial":
            resample_indices = rng.choice(
                np.arange(data.sizes[combined_dim]),
                size=num_samples,
                p=weights,
                replace=replace,
            )
        elif resampling_method == "stratified":
            if weights is None:
                raise ValueError("Weights must be provided for stratified resampling")
            resample_indices = _stratified_resample(weights, rng)

        data = data.isel({combined_dim: resample_indices})

    return data


def _stratified_resample(weights, rng):
    """Stratified resampling."""
    N = len(weights)
    single_uniform = (rng.random(N) + np.arange(N)) / N
    indexes = np.zeros(N, dtype=int)
    cum_sum = np.cumsum(weights)

    i, j = 0, 0
    while i < N:
        if single_uniform[i] < cum_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1

    return indexes


def dataset_to_dataarray(ds, sample_dims=None, labeller=None):
    """Convert a Dataset to a stacked DataArray, using a labeller to set coordinate values.

    Parameters
    ----------
    ds : Dataset
    sample_dims : sequence of hashable, optional
    labeller : labeller, optional

    Returns
    -------
    DataArray

    Examples
    --------
    Convert the posterior group into a stacked and labelled dataarray:

    .. jupyter-execute::

        import xarray as xr
        from arviz_base import load_arviz_data, dataset_to_dataarray
        xr.set_options(display_expand_data=False)

        idata = load_arviz_data("centered_eight")
        dataset_to_dataarray(idata.posterior.dataset)
    """
    if labeller is None:
        labeller = BaseLabeller()
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]

    labeled_stack = ds.to_stacked_array("label", sample_dims=sample_dims)
    labels = [
        labeller.make_label_flat(var_name, sel, isel)
        for var_name, sel, isel in xarray_sel_iter(ds, skip_dims=set(sample_dims))
    ]
    indexes = [idx_name for idx_name in labeled_stack.xindexes if idx_name not in sample_dims]
    labeled_stack = labeled_stack.drop_indexes(indexes).assign_coords(label=labels)
    for idx_name in indexes:
        if idx_name == "label":
            continue
        labeled_stack = labeled_stack.set_xindex(idx_name)
    return labeled_stack


def dataset_to_dataframe(ds, sample_dims=None, labeller=None, multiindex=False):
    """Convert a Dataset to a DataFrame via a stacked DataArray, using a labeller.

    Parameters
    ----------
    ds : Dataset
    sample_dims : sequence of hashable, optional
    labeller : labeller, optional
    multiindex : bool, default False

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    The output will have whatever is uses as `sample_dims` as the columns of
    the DataFrame, so when these are much longer we might want to transpose the
    output:

    .. jupyter-execute::

        from arviz_base import load_arviz_data, dataset_to_dataframe
        idata = load_arviz_data("centered_eight")
        dataset_to_dataframe(idata.posterior.dataset)

    The default is to only return a single index, with the labels or tuples of coordinate
    values in the stacked dimensions. To keep all data from all coordinates as a multiindex
    use ``multiindex=True``

    .. jupyter-execute::

        dataset_to_dataframe(idata.posterior.dataset, multiindex=True)

    The only restriction on `sample_dims` is that it is present in all variables
    of the dataset. Consequently, we can compute statistical summaries,
    concatenate the results into a single dataset creating a new dimension.

    .. jupyter-execute::

        import xarray as xr

        dims = ["chain", "draw"]
        post = idata.posterior.dataset
        summaries = xr.concat(
            (
                post.mean(dims).expand_dims(summary=["mean"]),
                post.median(dims).expand_dims(summary=["median"]),
                post.quantile([.25, .75], dim=dims).rename(
                    quantile="summary"
                ).assign_coords(summary=["1st quartile", "3rd quartile"])
            ),
            dim="summary"
        )
        summaries

    Then convert the result into a DataFrame for ease of viewing.

    .. jupyter-execute::

        dataset_to_dataframe(summaries, sample_dims=["summary"]).T

    Note that if all summaries were scalar, it would not be necessary to use
    :meth:`~xarray.Dataset.expand_dims` or renaming dimensions, using
    :meth:`~xarray.Dataset.assign_coords` on the result to label the newly created
    dimension would be enough. But using this approach we already generate a dimension
    with coordinate values and can also combine non scalar summaries.
    """
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    da = dataset_to_dataarray(ds, sample_dims=sample_dims, labeller=labeller)
    sample_dim = sample_dims[0]
    if len(sample_dims) > 1:
        da = da.stack(sample=sample_dims)
        sample_dim = "sample"
    if multiindex:
        idx_dict = {
            idx_name: da[idx_name].to_numpy()
            for idx_name in da.xindexes
            if sample_dim in da[idx_name].dims
        }
        sample_idx = pd.MultiIndex.from_arrays(list(idx_dict.values()), names=list(idx_dict.keys()))
        idx_dict = {
            idx_name: da[idx_name].to_numpy()
            for idx_name in da.xindexes
            if "label" in da[idx_name].dims
        }
        label_idx = pd.MultiIndex.from_arrays(list(idx_dict.values()), names=list(idx_dict.keys()))
    else:
        sample_idx = da[sample_dim]
        label_idx = da["label"]
    df = pd.DataFrame(
        da.transpose(sample_dim, "label").to_numpy(), columns=label_idx, index=sample_idx
    )
    if not multiindex:
        df.columns.name = "label"
        df.index.name = sample_dim
    return df


def explode_dataset_dims(ds, dim, labeller=None):
    """Explode dims of a dataset so each slice along them becomes its own variable.

    Parameters
    ----------
    ds : Dataset
    dim : hashable or sequence of hashable
        Dimension or dimensions along which slices to be stored as independent variables should
        be defined.
    labeller : labeller, optional
        Instance of a labeller class used to label the slices generated when exploding along `dim`.
        The method ``make_label_flat`` is used.

    Returns
    -------
    Dataset
        The dataset with all variables that have `dim` exploded into the respective slices
        as new variables.

    Examples
    --------
    In some cases, instead of ``theta`` as a ``(..., school)`` shape variable we'll want
    independent variables for each slice:

    .. jupyter-execute::

        from arviz_base import load_arviz_data, explode_dataset_dims
        import xarray as xr

        idata = load_arviz_data("centered_eight")
        explode_dataset_dims(idata.posterior.dataset, "school")
    """
    if isinstance(dim, str):
        dim = [dim]
    if labeller is None:
        labeller = BaseLabeller()
    return xr.Dataset(
        {
            labeller.make_label_flat(var_name, sel, isel): ds[var_name].sel(sel, drop=True)
            for var_name, sel, isel in xarray_sel_iter(
                ds, skip_dims={d for d in ds.dims if d not in dim}
            )
        }
    )


def references_to_dataset(references, ds, sample_dims=None):
    """Generate an :class:`~xarray.Dataset` compabible with `ds` from `references`.

    Cast common formats to provide references to a compatible Dataset.
    This function does not aim to be exhaustive, anything somewhat peculiar or complex
    will probably be better off building a Dataset manually instead.

    Parameters
    ----------
    references : scalar, 1D array-like, dict, DataArray, Dataset
        References to cast into a compatible dataset.

        * scalar inputs are interpreted as a reference line in each variable+coordinate not in
          `sample_dims` combination.
        * array-like inputs are interpreted as multiple reference lines in each variable+coordinate
          not in `sample_dims` combination. All subset having the same references
          and all references linked to every subset.
        * dict inputs are interpreted as array-like with each array matched to the variable
          corresponding to that dictionary key.
        * DataArray inputs are interpreted as an array-like if unnamed or as a single key
          dictionary if named.
        * Dataset inputs are returned as is but won't raise an error.

    ds : Dataset
        Dataset containing the data `references` should be compatible with.
    sample_dims : iterable of hashable, optional
        Sample dimensions in `ds`. The dimensions in the output will be the dimensions
        in `ds` minus `sample_dims` plus optionally a "ref_line_dim" for non-scalar references.

    Returns
    -------
    Dataset
        A Dataset with a subset of the variables, dimensions and coordinate names in `ds`,
        with only an extra dimension "ref_line_dim" added when multiple references are requested
        for one or some of the variables.

    See Also
    --------
    xarray.Dataset : Dataset constructor

    Examples
    --------
    Generate a reference dataset with 0 compatible with the centered eight example data:

    .. jupyter-execute::

        from arviz_base import load_arviz_data, references_to_dataset
        idata = load_arviz_data("centered_eight")
        references_to_dataset(0, idata.posterior.dataset)

    Generate a reference dataset with different references for each variable:

    .. jupyter-execute::

        references_to_dataset({"mu": -1, "tau": 1, "theta": 0}, idata.posterior.dataset)

    Or a similar case but with different number of references for each variable:

    .. jupyter-execute::

        ref_ds = references_to_dataset(
            {"mu": [-1, 0, 1], "tau": [1, 10], "theta": 0},
            idata.posterior.dataset
        )
        ref_ds

    Once we have a compatible dataset, we can for example compute the probability
    of the samples being above the reference value(s):

    .. jupyter-execute::

        (idata.posterior.dataset > ref_ds).mean()
    """
    # quick exit if dataset input
    if isinstance(references, xr.Dataset):
        return references
    # process argument defaults
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]

    # start covering cases, for dataarray, if its name is a variable convert to dataset
    # if it has no name treat is an array-like
    if isinstance(references, xr.DataArray):
        name = references.name
        if name is not None:
            if name not in ds.data_vars:
                raise ValueError(
                    "Input is a named DataArray whose name doesn't match any variable in `ds`. "
                    "Either use an unamed DataArray or ndarray or make sure the name matches."
                )
            return references.to_dataset()
        references = references.values
    # for scalars generate a dataset with requested shape full of reference value
    # check for numerical scalar following advise from
    # https://numpy.org/doc/2.2/reference/generated/numpy.isscalar.html
    if isinstance(references, Number):
        aux_ds = ds if sample_dims is None else ds.isel({dim: 0 for dim in sample_dims})
        return xr.full_like(aux_ds, references, dtype=np.array(references).dtype)
    # for array-like convert to dict so it is handled later on
    if isinstance(references, list | tuple | np.ndarray):
        if np.ndim(references) > 1:
            raise ValueError(
                "Only 1D arrays are allowed. To generate more complex reference datasets "
                "the xarray.Dataset constructor should be used."
            )
        references = {var_name: references for var_name in ds.data_vars}
    if isinstance(references, dict):
        ref_dict = {}
        for var_name, da in ds.items():
            if var_name not in references:
                continue
            ref_values = np.atleast_1d(references[var_name])
            if np.ndim(ref_values) > 1:
                raise ValueError(
                    f"Only 1D arrays are allowed but the values for {var_name} variable have "
                    "more dimensions. To generate more complex reference datasets the "
                    "xarray.Dataset constructor should be used."
                )
            sizes = {dim: length for dim, length in da.sizes.items() if dim not in sample_dims}
            ref_dict[var_name] = xr.DataArray(
                np.full(list(sizes.values()) + [len(ref_values)], ref_values),
                dims=list(sizes) + ["ref_line_dim"],
                coords={"ref_line_dim": np.arange(len(ref_values))}
                | {
                    coord_name: coord_da
                    for coord_name, coord_da in da.coords.items()
                    if coord_da.dims[0] not in sample_dims
                },
            )
        return xr.Dataset(ref_dict)
    raise TypeError("Unrecognized input type for `references`")
