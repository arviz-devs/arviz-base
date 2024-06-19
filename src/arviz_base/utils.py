"""General utilities."""

import re
import warnings

import numpy as np


def _check_tilde_start(x):
    return bool(isinstance(x, str) and x.startswith("~"))


def _var_names(var_names, data, filter_vars=None):
    """Handle var_names input across arviz.

    Parameters
    ----------
    var_names : str, list, or None
    data : xarray.Dataset or sequence of xarray.Dataset
    filter_vars : {None, "like", "regex"}, optional, default=None
        If `None` (default), interpret var_names as the real variables names. If "like",
        interpret var_names as substrings of the real variables names. If "regex",
        interpret var_names as regular expressions on the real variables names. A la
        `pandas.filter`.

    Returns
    -------
    var_name : list or None
    """
    if filter_vars not in {None, "like", "regex"}:
        raise ValueError(
            f"'filter_vars' can only be None, 'like', or 'regex', got: '{filter_vars}'"
        )

    if var_names is not None:
        if isinstance(data, list | tuple):
            all_vars = []
            for dataset in data:
                dataset_vars = list(dataset.data_vars)
                for var in dataset_vars:
                    if var not in all_vars:
                        all_vars.append(var)
        else:
            all_vars = list(data.data_vars)

        all_vars_tilde = [var for var in all_vars if _check_tilde_start(var)]
        if all_vars_tilde:
            warnings.warn(
                "ArviZ treats '~' as a negation character for variable selection. "
                f"Your model has variables names starting with '~', {', '.join(all_vars_tilde)}. "
                "Please double check your results to ensure all variables are included"
            )

        try:
            var_names = _subset_list(var_names, all_vars, filter_items=filter_vars, warn=False)
        except KeyError as err:
            msg = " ".join(("var names:", f"{err}", "in dataset"))
            raise KeyError(msg) from err
    return var_names


def _subset_list(subset, whole_list, filter_items=None, warn=True):
    """Handle list subsetting (var_names, groups...) across arviz.

    Parameters
    ----------
    subset : str, list, or None
    whole_list : list
        List from which to select a subset according to subset elements and
        filter_items value.
    filter_items : {None, "like", "regex"}, optional
        If `None` (default), interpret `subset` as the exact elements in `whole_list`
        names. If "like", interpret `subset` as substrings of the elements in
        `whole_list`. If "regex", interpret `subset` as regular expressions to match
        elements in `whole_list`. A la `pandas.filter`.

    Returns
    -------
    list or None
        A subset of ``whole_list`` fulfilling the requests imposed by ``subset``
        and ``filter_items``.
    """
    if subset is not None:
        if isinstance(subset, str):
            subset = [subset]

        whole_list_tilde = [item for item in whole_list if _check_tilde_start(item)]
        if whole_list_tilde and warn:
            warnings.warn(
                "ArviZ treats '~' as a negation character for selection. There are "
                f"elements in `whole_list` starting with '~', {', '.join(whole_list_tilde)}. "
                "Please double check your results to ensure all elements are included"
            )

        excluded_items = [
            item[1:] for item in subset if _check_tilde_start(item) and item not in whole_list
        ]
        filter_items = str(filter_items).lower()
        if excluded_items:
            not_found = []

            if filter_items in {"like", "regex"}:
                for pattern in excluded_items[:]:
                    excluded_items.remove(pattern)
                    if filter_items == "like":
                        real_items = [real_item for real_item in whole_list if pattern in real_item]
                    else:
                        # i.e filter_items == "regex"
                        real_items = [
                            real_item for real_item in whole_list if re.search(pattern, real_item)
                        ]
                    if not real_items:
                        not_found.append(pattern)
                    excluded_items.extend(real_items)
            not_found.extend([item for item in excluded_items if item not in whole_list])
            if not_found:
                warnings.warn(
                    f"Items starting with ~: {not_found} have not been found and will be ignored"
                )
            subset = [item for item in whole_list if item not in excluded_items]

        elif filter_items == "like":
            subset = [item for item in whole_list for name in subset if name in item]
        elif filter_items == "regex":
            subset = [item for item in whole_list for name in subset if re.search(name, item)]

        existing_items = np.isin(subset, whole_list)
        if not np.all(existing_items):
            raise KeyError(f"{np.array(subset)[~existing_items]} are not present")

    return subset


def _get_coords(data, coords):
    """Subselects xarray DataSet or DataArray object to provided coords. Raises exception if fails.

    Parameters
    ----------
    data : DataArray, Dataset or list of DataArray or Dataset
        Xarray objects to be subsetted. It can be a list or tuple, in which
        case all xarray objects whithin the iterable will be subsetted.
    coords : dict of {hashable: array_like}
        Dictionary specifying the subset to select. Passed to
        :meth:`xarray.Dataset.sel` or :meth:`xarray.DataArray.sel`
        depending on the input.

    Raises
    ------
    ValueError
        If coord values name are not available in data

    KeyError
        If dimension names are not available in data

    Returns
    -------
    data : Dataset or DataArray
        Return type is of the same type as the input
    """
    if not isinstance(data, list | tuple):
        try:
            return data.sel(**coords)

        except ValueError as err:
            invalid_coords = set(coords.keys()) - set(data.coords.keys())
            raise ValueError(f"Coords {invalid_coords} are invalid coordinate keys") from err

        except KeyError as err:
            raise KeyError(
                "Coords should follow mapping format {{coord_name:[dim1, dim2]}}. "
                "Check that coords structure is correct and"
                f" dimensions are valid. {err}"
            ) from err
    if not isinstance(coords, list | tuple):
        coords = [coords] * len(data)
    data_subset = []
    for idx, (datum, coords_dict) in enumerate(zip(data, coords)):
        try:
            data_subset.append(_get_coords(datum, coords_dict))
        except ValueError as err:
            raise ValueError(f"Error in data[{idx}]: {err}") from err
        except KeyError as err:
            raise KeyError(f"Error in data[{idx}]: {err}") from err
    return data_subset
