"""Generalistic converters."""

import numpy as np
import xarray as xr
from datatree import DataTree, open_datatree

from arviz_base.base import dict_to_dataset
from arviz_base.rcparams import rcParams
from arviz_base.utils import _var_names

__all__ = ["convert_to_datatree", "convert_to_dataset", "extract"]


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


def extract(
    data,
    group="posterior",
    combined=True,
    sample_dims=None,
    var_names=None,
    filter_vars=None,
    num_samples=None,
    weights=None,
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
    combined : bool, optional
        Combine `sample_dims` dimensions into ``sample``. Won't work if
        a dimension named ``sample`` already exists.
    sample_dims : iterable of hashable, optional
        List of dimensions that should be considered sampling dimensions.
        Random subsets and potential stacking if ``combine=True`` happen
        over these dimensions only. Defaults to ``rcParams["data.sample_dims"]``.
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
        Extract only a subset of the samples. Only valid if ``combined=True``
    weights : array-like, optional
        Extract a weighted subset of the samples. Only valid if ``combined=True`` and
        ``num_samples`` is not ``None``.
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
    if num_samples is not None and not combined:
        raise ValueError("num_samples is only compatible with combined=True")
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
    if combined:
        if sample_dims is None:
            sample_dims = rcParams["data.sample_dims"]
        data = data.stack(sample=sample_dims)

    if weights is not None or num_samples is not None:
        if random_seed is None:
            rng = np.random.default_rng()
        elif isinstance(random_seed, (int, np.integer)):
            rng = np.random.default_rng(random_seed)
        elif isinstance(random_seed, np.random.Generator):
            rng = random_seed
        else:
            raise ValueError(f"Invalid random_seed value: {random_seed}")
        
        replace = True if weights is not None else False
        resample_indices = rng.choice(np.arange(len(data["sample"])), size=num_samples, p=weights, replace=replace)
        data = data.isel(sample=resample_indices)

    return data
