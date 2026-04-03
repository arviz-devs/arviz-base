"""ArviZ input validation utilities."""

import typing
import warnings

from arviz_base.rcparams import defaultParams, rcParams


def validate_sample_dims(sample_dims, data=None):
    """Validate `sample_dims` argument in ArviZ functions.

    Parameters
    ----------
    sample_dims : str or sequence of str or None
        Common input to ArviZ functions.
    data : Dataset or DataTree, optional
        Data given as input along `sample_dims`.

    Returns
    -------
    sample_dims : list of str
        Validated `sample_dims`. When `sample_dims` is ``None`` the default is taken from:

        * If `data` is given, it defaults to the "sample_dims" key in its attributes
        * Otherwise the :py:data:`data.sample_dims` rcParam is used.

    Raises
    ------
    ValueError
        If some of the dimensions are not present in `data`.
        This check is skipped if `data` is not given.
    """
    if data is None:
        if sample_dims is None:
            sample_dims = rcParams["data.sample_dims"]
        if isinstance(sample_dims, str):
            return [sample_dims]
        return list(sample_dims)
    if sample_dims is None and "sample_dims" in data.attrs:
        return data.attrs["sample_dims"]
    sample_dims = validate_sample_dims(sample_dims)
    missing_sample_dims = [dim for dim in sample_dims if dim not in data.dims]
    if missing_sample_dims:
        raise ValueError(
            f"Invalid value for sample_dims. Got {sample_dims} but "
            f"present dimensions are {data.dims}."
        )
    return sample_dims


def validate_dims_chain_draw_axis(dims, data=None):
    """Validate `dims` argument for functions that use chain_axis and draw_axis.

    In such cases, dims can have length 1 or 2 depending on there being a chain dimension.

    Returns
    -------
    dims : list
        List of dimensions
    chain_axis : int or None
        Positional index for chain dimension
    draw_axis : int
        Positional index for draw dimension
    """
    dims = validate_sample_dims(dims, data=data)
    draw_axis = -1
    if len(dims) == 1:
        chain_axis = None
    elif len(dims) == 2:
        chain_axis = -2
    else:
        raise ValueError("dims can only have 1 or 2 elements")
    return dims, chain_axis, draw_axis


def validate_dict_argument(dict_arg, func_arg=None, *, valid_keys=None):
    """Validate dict arguments of plotting functions.

    Parameters
    ----------
    dict_arg : mapping of {str : any} or None
        Dictionary argument of plotting functions: "visuals", "aes_by_visuals" or "stats"
    func_arg : tuple of (callable, str) or None
        Tuple with the function and the argument name for which to perform the validation.
        When given, `valid_keys` is taken from the type hint of the given function and argument.
        The type hint must therefore be of the form ``Mapping[Literal[...], ...]``
    valid_keys : sequence of str or None
        Collection of all the valid keys `dict_args` can have.

    Returns
    -------
    dict_arg : dict of {str : any}
    """
    if func_arg is None and valid_keys is None:
        raise ValueError("Either 'func' or 'valid_keys' argument must be given.")
    if func_arg is not None and valid_keys is not None:
        warnings.warn("Both 'func_arg' and 'valid_keys' provided, 'valid_keys' will be ignored")
    if dict_arg is None:
        dict_arg = {}
    else:
        dict_arg = dict_arg.copy()
    msg = ""
    if func_arg is not None:
        func, arg_name = func_arg
        msg = f"In argument {arg_name} of {func.__name__}\n"
        valid_keys = typing.get_args(typing.get_args(typing.get_type_hints(func)[arg_name])[0])
    extra_keys = [key for key in dict_arg if key not in valid_keys]
    if extra_keys:
        msg += f"Found keys {extra_keys} but valid keys are {valid_keys}"
        raise ValueError(msg)
    return dict_arg


def validate_ci_prob(prob):
    """Validate `prob`/`ci_prob` argument.

    Parameters
    ----------
    prob : float or None
        The probability to validate. It can also be None in which case
        the rcParam value for "stats.ci_prob" is used.

    Returns
    -------
    float
    """
    if prob is None:
        prob = rcParams["stats.ci_prob"]
    return validate_prob(prob)


def validate_prob(prob, allow_0=False):
    r"""Validate required `prob` argument.

    Parameters
    ----------
    prob : float
    allow_0 : bool, default False
        Whether to restrict to :math:`(0, 1]` for probability values
        or, when True, include 0 to allow :math:`[0, 1]` as valid probabilities.

    Returns
    -------
    float
    """
    if allow_0 and not 1 >= prob >= 0:
        raise ValueError(f"The value of prob should be in the interval [0, 1] but got {prob}")
    if not allow_0 and not 1 >= prob > 0:
        raise ValueError(f"The value of prob should be in the interval (0, 1] but got {prob}")
    return prob


def validate_or_use_rcparam(arg_in, rckey):
    """Validate an arbitrary argument that defaults to an rcParam.

    Parameters
    ----------
    arg_in : any
        Input value for argument that defaults to `rckey`
    rckey : str
        The rcParams key from which to take the default value when `arg_in`
        is ``None`` and the validation function otherwise.

    Returns
    -------
    arg_out : any
    """
    if arg_in is None:
        return rcParams[rckey]
    return defaultParams[rckey][1](arg_in)
