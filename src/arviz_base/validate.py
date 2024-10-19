import inspect
import itertools
from collections.abc import Hashable, Iterable
from functools import wraps
import typing

from arviz_base.rcparams import rcParams


def _sample_dims(value, dt, group, func_name):
    groups = [g for g in map(lambda g: g.replace("/", ""), dt.groups) if g]
    dims = {g: list(getattr(dt, g).dims) for g in groups}

    if value is None:
        value = rcParams["data.sample_dims"]
    if not isinstance(value, Iterable):
        raise TypeError(f"sample_dims of {func_name!r} must be the names of dimensions.")
    if not all(isinstance(v, Hashable) for v in value):
        raise TypeError(
            f"sample_dims of {func_name!r} must be names of dimensions, or tuples of dimensions."
        )
    if isinstance(value, list):
        if not all(isinstance(v, str) for v in value):
            raise TypeError(f"sample_dims of {func_name!r} must be the name of dimension.")
    if not isinstance(value, list):
        value = [value]
    exploded_value = list(itertools.chain.from_iterable(value))
    if not set(exploded_value).issubset(set(dims[group])):
        # import ipdb
        #
        # ipdb.set_trace()
        raise ValueError(
            f"The DataTree does not contain the given sample_dim for the given group '{group}'. "
            f"Available sample_dim are {dims[group]}"
        )
    return value


def _group(value, dt, func_name):
    if not isinstance(value, str):
        raise TypeError(f"group of {func_name!r} must be a string.")
    groups = [group for group in map(lambda group: group.replace("/", ""), dt.groups) if group]
    if value not in groups:
        raise TypeError(
            f"group of {func_name!r} must be the name of a group in the DataTree. "
            f"Available groups are: {list(groups)}"
        )
    return value


def extract(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        parameters = {
            k: v.default
            for k, v in inspect.signature(func).parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        parameters.update(kwargs)

        # Validate group
        dt = parameters["data"]
        group = parameters["group"]
        kwargs["group"] = _group(group, dt, func.__name__)

        # Validate sample_dims
        kwargs["sample_dims"] = _sample_dims(parameters["sample_dims"], dt, group, func.__name__)

        # Switch combined based on the length of sample_dims
        if len(kwargs["sample_dims"]) == 1:
            kwargs["combined"] = True

        # Handle num_samples
        if parameters["num_samples"] is not None and not parameters["combined"]:
            raise ValueError(
                "num_samples is only compatible with combined=True or length 1 sample_dims"
            )

        # Handle weights
        if parameters["weights"] is not None and parameters["num_samples"] is None:
            raise ValueError("weights are only compatible with num_samples")

        return func(*args, **kwargs)

    return wrapper
