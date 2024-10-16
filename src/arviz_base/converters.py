"""Generalistic converters."""

import numpy as np
import pandas as pd
import xarray as xr
from datatree import DataTree, open_datatree

from arviz_base.base import dict_to_dataset
from arviz_base.labels import BaseLabeller
from arviz_base.rcparams import rcParams
from arviz_base.sel_utils import xarray_sel_iter
from arviz_base.utils import _var_names

__all__ = [
    "convert_to_datatree",
    "convert_to_dataset",
    "extract",
    "to_labelled_stacked_da",
    "to_labelled_stacked_df",
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
            f'Can only convert {", ".join(allowable_types)} to InferenceData, '
            f"not {obj.__class__.__name__}"
        )

    return DataTree.from_dict(d={group: dataset})


def convert_to_dataset(obj, *, group="posterior", **kwargs):
    """Convert a supported object to an xarray dataset.

    This function is idempotent, in that it will return xarray.Dataset functions
    unchanged. Raises `ValueError` if the desired group can not be extracted.

    Note this goes through a DataTree object via :func:`convert_to_datatree`.
    See its docstring for more details.

    Parameters
    ----------
    obj
        A supported object to convert to InferenceData.
    group : str, default "posterior"
        If `obj` is a dict or numpy array, assigns the resulting xarray
        dataset to this group.
    **kwargs : dict, optional
        Keyword arguments passed to :func:`convert_to_datatree`

    Returns
    -------
    xarray.Dataset
        New mutable dataset. See :meth:`datatree.DataTree.to_dataset` for more details.

    Raises
    ------
    ValueError
        If `obj` can't be converted to a DataTree from which to extract the
        `group` Dataset.

    See Also
    --------
    dict_to_dataset
        Convert a dictionary of arrays to a :class:`xarray.Dataset` following ArviZ conventions.
    """
    if isinstance(obj, DataTree) and obj.name == group:
        return obj.to_dataset()
    inference_data = convert_to_datatree(obj, group=group, **kwargs)
    dataset = getattr(inference_data, group, None)
    if dataset is None:
        raise ValueError(
            f"Can not extract {group} from {obj}! See docs for other " "conversion utilities."
        )
    return dataset.to_dataset()


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


def to_labelled_stacked_da(ds, sample_dims=None, labeller=None):
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

    .. jupyter-execute::

        import xarray as xr
        from arviz_base import load_arviz_data, to_labelled_stacked_da
        xr.set_options(display_expand_data=False)

        idata = load_arviz_data("centered_eight")
        to_labelled_stacked_da(idata.posterior.ds)
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


def to_labelled_stacked_df(ds, sample_dims=None, labeller=None, multiindex=False):
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

        from arviz_base import load_arviz_data, to_labelled_stacked_df
        idata = load_arviz_data("centered_eight")
        to_labelled_stacked_df(idata.posterior.ds).T

    The default is to only return a single index, with the labels or tuples of coordinate
    values in the stacked dimensions. To keep all data from all coordinates as a multiindex
    use ``multiindex=True``

    .. jupyter-execute::

        to_labelled_stacked_df(idata.posterior.ds, multiindex=True).T

    The only restriction on `sample_dims` is that it is present in all variables
    of the dataset. Consequently, we can compute statistical summaries,
    concatenate the results into a single dataset creating a new dimension.

    .. jupyter-execute::

        import xarray as xr

        dims = ["chain", "draw"]
        post = idata.posterior.ds
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

        to_labelled_stacked_df(summaries, sample_dims=["summary"])

    Note that if all summaries were scalar, it would not be necessary to use
    :meth:`~xarray.Dataset.expand_dims` or renaming dimensions, using
    :meth:`~xarray.Dataset.assign_coords` on the result to label the newly created
    dimension would be enough. But using this approach we already generate a dimension
    with coordinate values and can also combine non scalar summaries.
    """
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    da = to_labelled_stacked_da(ds, sample_dims=sample_dims, labeller=labeller)
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
        columns = pd.MultiIndex.from_arrays(list(idx_dict.values()), names=list(idx_dict.keys()))
        idx_dict = {
            idx_name: da[idx_name].to_numpy()
            for idx_name in da.xindexes
            if "label" in da[idx_name].dims
        }
        index = pd.MultiIndex.from_arrays(list(idx_dict.values()), names=list(idx_dict.keys()))
    else:
        columns = da[sample_dim]
        index = da["label"]
    df = pd.DataFrame(da.transpose("label", sample_dim).to_numpy(), columns=columns, index=index)
    if not multiindex:
        df.index.name = "label"
        df.columns.name = sample_dim
    return df
