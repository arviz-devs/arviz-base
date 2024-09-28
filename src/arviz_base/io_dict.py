"""Dictionary specific conversion code."""

import warnings

from datatree import DataTree

from arviz_base.base import dict_to_dataset
from arviz_base.rcparams import rcParams


def from_dict(
    data,
    *,
    name=None,
    sample_dims=None,
    save_warmup=None,
    index_origin: int | None = None,
    coords=None,
    dims=None,
    pred_dims=None,
    pred_coords=None,
    check_conventions=True,
    attrs=None,
):
    """Convert nested dictionary into a DataTree.

    It uses :func:`dict_to_dataset` to convert a nested dictionary to a DataTree
    using ArviZ conventions.

    Parameters
    ----------
    data : dict of {str : dict of {hashable: array_like}}
        Dictionary to convert to DataTree. It must be a nested dictionary.
        The keys of the outer dictionary are group names, the values
        of the outer dictionary must be dictionaries themselves with the
        variables that should be stored in that group. These inner dictionaries
        are passed to :func:`dict_to_dataset`.

        Depending on the group name, the arguments used when calling `dict_to_dataset`
        can take different arguments from their defaults:

        * If ``data`` is a substring of the group name, ``sample_dims`` will be
          set to an empty list: ``[]``
        * If ``predictions`` is a substring of the group name, `pred_coords`
          and `pred_dims` will be passed to :func:`dict_to_dataset` as ``coords``
          and ``dims``.
        * If ``log_likelihood`` is a substring of the group name,
          ``skip_event_dims`` is set to ``True``.

    name : str, optional
        Name of the DataTree root node. This is used as the DataTree name by
        ArviZ.
    sample_dims : iterable of hashable, optional
        Dimensions that should be assumed to be present in _all_ variables.
        If missing, they will be added as the dimensions corresponding to the
        leading axis.
    save_warmup : bool, optional
        Save warmup iterations DataTree. If not defined, use default
        defined by the rcParams. When set to ``False``, groups in `data` that have
        ``warmup`` as substring will be ignored.
    index_origin : int, optional
        Start value to use default integer ids for dimensions without provided
        coordinate values. Defaults to ``data.index_origin``.
    coords : dict of {hashable : list}, optional
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict of {hashable : list of hashable}, optional
        A mapping from variable names to a list of dimension names for the variable.
    pred_dims : dict of {str : list of str}, optional
        A mapping from variables to a list of coordinate names for predictions.
    pred_coords : dict of {str : list}, optional
        A mapping from variables to a list of coordinate values for predictions.
    check_conventions : bool, default True
        Check some ArviZ conventions on dimension meaning.
    attrs : dict of {str : dict}, optional
        A dictionary containing attributes for different groups. Its keys should
        match keys in `data`, with the exception of "/" which is used to set global
        attributes.

    Returns
    -------
    DataTree

    See Also
    --------
    datatree.DataTree.from_dict
    """
    d_ds = {}
    if attrs is None:
        attrs = {}

    bad_attr_groups = [group for group in attrs if (group not in data) and (group != "/")]
    if bad_attr_groups:
        warnings.warn(
            f"Found groups in `attrs` ({bad_attr_groups}) that are not present in `d`. "
            "They will be ignored",
            UserWarning,
        )

    if save_warmup is None:
        save_warmup = rcParams["data.save_warmup"]
    for group, group_data in data.items():
        if save_warmup is False and "warmup" in group:
            continue
        d_ds[group] = dict_to_dataset(
            group_data,
            attrs=attrs.get(group, None),
            coords=pred_coords if "predictions" in group else coords,
            dims=pred_dims if "predictions" in group else dims,
            sample_dims=[] if "data" in group else sample_dims,
            index_origin=index_origin,
            skip_event_dims="log_likelihood" in group,
            check_conventions=check_conventions,
        )
    dt = DataTree.from_dict(d_ds, name=name)
    if "/" in attrs:
        dt.attrs = attrs["/"]
    return dt
